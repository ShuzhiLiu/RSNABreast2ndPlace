# RSNABreast2ndPlace
## Competition: 
RSNA Screening Mammography Breast Cancer Detection  
Find breast cancers in screening mammograms

[Solution summary](solution_summary.md)

## Files

* `train_withbox_split_clean_encode.csv`:
    * added bbox info to original `train.csv`
    * encoded `nan` to 255 and string type to int

## Train

1. Put `train_withbox_split_clean_encode.csv`
   to `/kaggle/input/rsna-breast-cancer-detection`
2. Use `train.py`

        python train.py /path_to_config

## External data

If you are interested in pretraining with external data, please follow these
steps:

1. Download the necessary data from the
   provided [link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/377790).
2. Add the relevant information of the downloaded data to the
   train_withbox_split_clean_encode.csv file. For any labels that are missing,
   please use the value 255.
