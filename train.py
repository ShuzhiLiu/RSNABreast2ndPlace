import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
from typing import Sequence, Union, List, Optional

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.datasets import CustomDataset
from mmcls.models.classifiers.image import ImageClassifier
from mmcls.registry import METRICS
from mmcls.registry import MODELS
from mmcls.structures.cls_data_sample import ClsDataSample
from mmcls.utils import register_all_modules
from mmcv.transforms import LoadImageFromFile, BaseTransform
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.dist import all_reduce as allreduce
from mmengine.evaluator import BaseMetric
from mmengine.registry import DATASETS
from mmengine.registry import TRANSFORMS
from mmengine.runner import Runner
from mmengine.structures import LabelData
from mmengine.utils import digit_version
from mmengine.utils import is_str
from mmengine.utils.dl_utils import TORCH_VERSION
from torch.nn.modules.loss import _Loss


def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = pydicom.dcmread(path)
    # pixel_sign = img[('0028', '1041')].value # Not this value
    # img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if img.PhotometricInterpretation == "MONOCHROME1":
        data = 255 - data
    return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)


# From https://github.com/Ezra-Yu/ACCV2022_FGIA_1st

@MODELS.register_module(force=True)
class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0,
                 eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg',
                             'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:,
                                                          None].clamp(
            min=self.eps)
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        allreduce(pos_grad)
        allreduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


@DATASETS.register_module(force=True)
class CsvGeneralDataset(CustomDataset):
    def __init__(self, ann_file, metainfo: Optional[dict] = None,
                 data_root: str = '', data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 lazy_init: bool = False, split=0, train=True,
                 label_key='',
                 **kwargs):
        self.split = split
        self.train_fold = train
        assert label_key
        self.label_key = label_key
        super().__init__(ann_file, metainfo, data_root, data_prefix, extensions,
                         lazy_init, **kwargs)

    def load_data_list(self):

        df1 = pd.read_csv(self.ann_file)
        if self.split >= 0:
            if self.train_fold:
                df1 = df1[df1['split'] != self.split]
            else:
                df1 = df1[df1['split'] == self.split]
        data_list = df1.to_dict("records")
        return data_list

    def get_gt_labels(self):
        gt_labels = np.array(
            [self.get_data_info(i)[self.label_key] for i in range(len(self))],
            dtype=np.int64)
        return gt_labels


@TRANSFORMS.register_module(force=True)
class LoadImageRSNABreastAux(LoadImageFromFile):
    def __init__(self, to_float32: bool = False, color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 ignore_empty: bool = False,
                 cropped=True,
                 file_key='',
                 label_key='',
                 img_prefix='',
                 extension='') -> None:
        super().__init__(to_float32, color_type, imdecode_backend,
                         file_client_args, ignore_empty)
        self.file_key = file_key
        self.label_key = label_key
        self.img_prefix = img_prefix
        self.extension = extension
        self.cropped = cropped

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        cache_folder = 'cropped_images' if self.cropped else 'cvt_images'
        cached_file = f"/kaggle/input/rsna-breast-cancer-detection/{cache_folder}/{results['image_id']}.jpg"
        if os.path.isfile(cached_file):
            img = cv2.imread(cached_file)
        else:
            filename = f"{results['patient_id']}/{results['image_id']}.dcm"
            filename = os.path.join(self.img_prefix, filename)
            img = load_dicom(filename)
            cvt_file = f"/kaggle/input/rsna-breast-cancer-detection/cvt_images/{results['image_id']}.jpg"
            if not os.path.isfile(cvt_file):
                cv2.imwrite(cvt_file, img)
            # crop image
            if self.cropped:
                x, y, w, h = results['x'], results['y'], results['w'], results[
                    'h']
                x2, y2 = x + w, y + h
                img = img[y:y2, x:x2, :]
                cv2.imwrite(cached_file, img)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # label part
        results['gt_label'] = int(results['cancer'])

        # aux label part
        view = results['view']
        BIRADS = results['BIRADS']
        invasive = results['invasive']
        difficulty = results['difficult_negative_case']
        implant = results['implant']
        age = results['age']
        density = results['density']
        aux_label = np.array(
            [view, BIRADS, invasive, difficulty, implant, age, density],
            dtype=np.float32)
        results['aux_label'] = aux_label

        return results


class MxDataSample(ClsDataSample):
    """
    without correct collate_fn the aux label will stay in cpu
    """

    def set_aux_label(
            self, value) -> 'ClsDataSample':
        label_data = getattr(self, '_aux_label', LabelData())
        label_data.label = torch.tensor(value, dtype=torch.float32)
        self.aux_label = label_data
        return self

    @property
    def aux_label(self):
        return self._aux_label

    @aux_label.setter
    def aux_label(self, value: LabelData):
        self.set_field(value, '_aux_label', dtype=LabelData)

    @aux_label.deleter
    def aux_label(self):
        del self._aux_label


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackMxInputs(BaseTransform):
    """Pack the inputs data for the classification.
    **Required Keys:**
    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)
    **Deleted Keys:**
    All keys in the dict.
    **Added Keys:**
    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmcls.structures.ClsDataSample`): The annotation
      info of the sample.
    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:
            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)
        else:
            warnings.warn(
                'Cannot get "img" in the input dict of `PackClsInputs`,'
                'please make sure `LoadImageFromFile` has been added '
                'in the data pipeline or images have been loaded in '
                'the dataset.')

        data_sample = MxDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)
        if 'aux_label' in results:
            data_sample.set_aux_label(results['aux_label'])

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (
                beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 100)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]


@METRICS.register_module(force=True)
class RSNAPFBeta(BaseMetric):

    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()

            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']

            result['pred_scores'] = pred_label['score']

            result['gt_score'] = gt_label['label']

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        # concat
        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_scores'][1] for res in results])
        target = target.squeeze().cpu().numpy().flatten()
        pred = pred.squeeze().cpu().numpy().flatten()
        assert len(target) == len(
            pred), f"target: {len(target)}, pred: {len(pred)}"

        f1, thres = optimal_f1(target, pred)

        result_metrics = dict()

        result_metrics['pf1'] = f1
        result_metrics['thres'] = thres

        return result_metrics


@MODELS.register_module(force=True)
class RSNAAuxCls(ImageClassifier):
    def __init__(self, backbone: dict, neck: Optional[dict] = None,
                 head: Optional[dict] = None, pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(backbone, neck, head, pretrained, train_cfg,
                         data_preprocessor, init_cfg)
        in_channels = head['in_channels']
        self.nn_view = nn.Linear(in_channels, 6)
        self.nn_BIRADS = nn.Linear(in_channels, 3)
        self.nn_difficulty = nn.Linear(in_channels, 2)
        self.nn_density = nn.Linear(in_channels, 4)
        self.nn_sigmoid = nn.Linear(in_channels, 3)

        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
        self.sigmoid_loss = torch.nn.BCEWithLogitsLoss()

        self.BIRADS_lossfn = SoftmaxEQLLoss(num_classes=3)
        self.diff_lossfn = SoftmaxEQLLoss(num_classes=2)
        self.density_lossfn = SoftmaxEQLLoss(num_classes=4)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[MxDataSample]) -> dict:
        feats = self.extract_feat(inputs)
        loss_cancer = self.head.loss(feats, data_samples)
        cancer_target = torch.stack(
            [i.gt_label.label for i in data_samples]).to(
            'cuda:0')
        aux_target = torch.stack([i.aux_label.label for i in data_samples]).to(
            'cuda:0')
        feats = feats[-1]
        aux_weight = 0.1

        loss_view = self.ce_loss(self.nn_view(feats),
                                 aux_target[:, 0].to(torch.long))
        loss_cancer.update({'loss_view': loss_view * aux_weight, })

        BIRADS_mask = aux_target[:, 1] < 255
        if torch.sum(BIRADS_mask) > 0:
            loss_BIRADS = self.BIRADS_lossfn(
                self.nn_BIRADS(feats)[BIRADS_mask, :],
                aux_target[:, 1][BIRADS_mask].to(torch.long))
            loss_cancer.update({'loss_BIRADS': loss_BIRADS * aux_weight, })

        sig_out = self.nn_sigmoid(feats)
        invasive_mask = torch.logical_and(aux_target[:, 2] < 255,
                                          cancer_target[:, 0] > 0)
        if torch.sum(invasive_mask) > 0:
            loss_invasive = self.sigmoid_loss(sig_out[:, 0][invasive_mask],
                                              aux_target[:, 2][invasive_mask])
            loss_cancer.update({'loss_invasive': loss_invasive * aux_weight, })

        difficulty_mask = torch.logical_and(aux_target[:, 3] < 255,
                                            cancer_target[:, 0] < 1)
        if torch.sum(difficulty_mask) > 0:
            loss_difficulty = self.diff_lossfn(
                self.nn_difficulty(feats)[difficulty_mask, :],
                aux_target[:, 3][difficulty_mask].to(torch.long))
            loss_cancer.update(
                {'loss_difficulty': loss_difficulty * aux_weight, })

        # implant_mask = aux_target[:, 4] < 255
        # if torch.sum(implant_mask) > 0:
        #     loss_implant = self.sigmoid_loss(sig_out[:, 1][implant_mask],
        #                                      aux_target[:, 4][implant_mask])
        #     loss_cancer.update({'loss_implant': loss_implant * aux_weight, })

        age_mask = aux_target[:, 5] < 255
        if torch.sum(age_mask) > 0:
            loss_age = self.sigmoid_loss(sig_out[:, 2][age_mask],
                                         aux_target[:, 5][age_mask] / 100)
            loss_cancer.update({'loss_age': loss_age * aux_weight, })
        density_mask = aux_target[:, 6] < 255
        if torch.sum(density_mask) > 0:
            loss_density = self.density_lossfn(
                self.nn_density(feats)[density_mask],
                aux_target[:, 6][density_mask].to(torch.long))
            loss_cancer.update({'loss_density': loss_density * aux_weight})

        return loss_cancer

    def psuedo_label(self, inputs, **kwargs):
        feats = self.extract_feat(inputs)
        feats = feats[-1]
        BIRADS = self.nn_BIRADS(feats).softmax(dim=1)
        Density = self.nn_density(feats).softmax(dim=1)
        return (BIRADS, Density)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
             'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    # register all modules in mmcls into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
