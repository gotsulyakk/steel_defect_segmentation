# Steel defect detection [test task]
Aim is to create simple binary segementation pipeline using [pytorch lightning](https://www.pytorchlightning.ai/), [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) and data from [severstal steel defect detection challenge](https://www.kaggle.com/c/severstal-steel-defect-detection/overview)

## Install
```sh
git clone https://github.com/gotsulyakk/test_task_steel_defect_detection.git
cd test_task_steel_defect_detection

pip install -r requirements.txt
```
## Data
[Original dataset](https://www.kaggle.com/c/severstal-steel-defect-detection/data) contains both labeled and unlabeled grayscale (ith 3 channels) images with defects (4 classes) and without defects. Goal of this task is to combine 4 classes into 1 and do binary segmentation. 

Data prepared with notebooks/data_prep.ipynb

- Original image resolution: 256x1600
- Image resolution during training: 256x416
- Number of images used in this task: 800 for train, 100 for validation, 100 for test set.

## Result
Model have **~50 IoU** on test set

## Training
You can run training command with:
```sh
python train.py --config {PATH/TO/CONFIG}
```
**Architecture**: I tried Unet and FPN architectures. FPN seems to work slight better (~1%). \
**Encoder**: resnext50_32x4d was ~2% better than efficientnetb3 in my few runs. \
**Loss function**: Dice \
**Opimizer**: AdamW(lr=0.0001)
**Metric**: IoU(threshold=0.5) \
**Number of epochs**: 30 \

## Inference example
You can run demo.py for inference:
```sh
python demo.py --image {PATH/TO/IMAGE} --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```
Random image from test set: \
Original image:
![Original image](https://github.com/gotsulyakk/test_task_steel_defect_detection/blob/main/data/inference/original.jpg)
Predicted mask:
![Predicted mask](https://github.com/gotsulyakk/test_task_steel_defect_detection/blob/main/data/inference/pred_mask.jpg)
Image with mask overlay:
![Image with mask](https://github.com/gotsulyakk/test_task_steel_defect_detection/blob/main/data/inference/masked.jpg)
Ground truth mask:
![Ground truth mask](https://github.com/gotsulyakk/test_task_steel_defect_detection/blob/main/data/inference/gt_mask.jpg)

## Further improvements
1. The most common - add more data
2. Try another architecture and/or another backbone
3. Try another loss function or combine them
4. Try another optimizer
5. Play with image resolution which require more computational power
5. Find best learning rate
4. If we don't want to "label" more data we can "pseudolabel" it and use for training 
5. Use postprocessing techniques such as dillation to improve masks quality

## Problems
- Data: in some cases without domain knowledge I couldn't figure out where is a defect on original image without ground truth
- We cannot just copy-paste kaggle notebooks since requirements of this task differ from requirements of the original competition
- It was quite interesting (and hard a bit) to figure out how to work with mask encoding in train.csv and how to combine it into one class
- I'm quite new to pytorch ligthtning so code can contains bugs and the code style may not follow best practices
