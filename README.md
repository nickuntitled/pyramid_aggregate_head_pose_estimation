# Pyramid-based structure for head pose estimation

![Architecture](https://i.ibb.co/wWrL7RG/1-Latest-Architecture.jpg)

This repository proposes HPE architecture, being in the landmark-free category. This study aims to build a pyramidal architecture to extract image features at multiple levels of details and then aggregate them to synergize advantages of multi-scale semantic information. The bottom layers contain edges and corners, and the top layers contains abstract features for classification.

In addition, our architecture has an objective to increase the attention in spatial and channel levels, which focuses on where and what to pay attention to each aggregated feature map. Hence, the performance of head pose classification and regression is improved by our design. 

## How to prepare

Download the dataset, and extract the dataset into datasets/< dataset name folder>.

#### 300W_LP

You have to download [300W_LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset from the author, and extract into datasets/300W_LP folder. The structure should be like this.

```
datasets
-- 300W_LP
---- AFW
---- LFPW
---- HELEN
---- IBUG
---- AFW_Flip
---- HELEN_Flip
---- LFPW_Flip
---- IBUG_Flip
```

#### AFLW2000

You have to download [AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset from the author, and extract into datasets/AFLW2000 folder. The structure should be like this.

```
datasets
-- AFLW2000
---- *.jpg and *.mat files
```

#### BIWI

You have to download the [prepared BIWI](https://drive.google.com/file/d/1T2VfY35hSVPF9uNzJteZQOnz_ADQqNqM/view?usp=sharing) dataset, and extract into datasets/BIWI folder. The structure should be like this.

```
datasets
-- BIWI
---- *.npz files
```

#### DAD-3DHeads

You have to download the [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads.git) dataset from the author, and extract into datasets/DAD folder. The structure should be like this.

```
datasets
-- DAD
---- train
------ annotations
------ images
```

## How to train

Train by using command below:

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epoch <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size>
```

#### DAD-3DHeads

Train on DAD first. (If you don't want to train, you can download pretrained from Google Drive for [EfficientNetV2-S](https://drive.google.com/file/d/17IbejEhOkUQ7UOQaHy_QVDvRDAr5owoI/view?usp=sharing) or [EfficientNetV2-M](https://drive.google.com/file/d/1URZwah8dfOPMuE1Wh4s1Hjg7M_poHuXH/view?usp=sharing).)

```
python train.py --dataset DAD --data_dir datasets/DAD/ --filename_list datasets/DAD/train.json --num_epoch 50 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string DAD --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt      
```

#### 300W_LP

Train on 300W_LP.

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt               
```

#### BIWI

Train on BIWI.

```
python train.py --dataset BIWI --data_dir datasets/BIWI/ --filename_list datasets/BIWI/biwi_train_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_test_list.txt     
```

## How to evaluate

Test on the desired datasets. How to do.

#### For train by 300W_LP

Test on AFLW2000

```
python test.py --dataset AFLW2000 --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 224 --crop 0
```

Test on BIWI

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 240 --crop_size 224 --crop 1
```

#### For train by BIWI

Test on BIWI

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_test_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 224 --crop_size 224 --crop 0
```

## Result

The evaluation results are shown below.

|                                 | yaw  | pitch | roll | mean |
|---------------------------------|------|-------|------|------|
| AFLW2000                        | 2.97 | 4.29  | 2.83 | 3.36 |
| BIWI                            | 3.51 | 4.31  | 2.71 | 3.51 |
| BIWI train and test (70 and 30) | 2.01 | 2.66  | 1.73 | 2.13 |