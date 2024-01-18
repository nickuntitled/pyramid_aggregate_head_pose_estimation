# Pyramid-based structure for head pose estimation

![Architecture](https://i.ibb.co/wWrL7RG/1-Latest-Architecture.jpg)

This repository proposes landmark-free HPE method. This is based on my internship project in NCCU (National Chung Cheng University), Taiwan. This study aims to build a pyramidal architecture to extract multi-level image features and then aggregate them to synergize advantages of multi-scale semantic information. The bottom layers contain edges and corners, and the top layers contains abstract features for classification.

In addition, our architecture has an objective to increase the attention in spatial and channel levels, which focuses on where and what to pay attention to each aggregated feature map. Hence, the performance of head pose classification and regression is improved by our design. 

## Pyramidal structure

Multi-scale image features (or multi-levels of details) are extracted from a designed pyramid structure after multiple layers of convolution blocks, where the higher layers produce smaller feature maps to response entire objects for abstract characteristics, and the lower layers produce larger feature maps to reveal detailed edges or corners information. As a result, gradual accumulation of pyramidal features synergizes different tasks for HPE improvement. This architecutre is based from EfficientNetV2-S, the CNN backbone, with the first five convolution blocks.

## Feature Pyramid Aggregation Structure

![FPAS](https://i.ibb.co/cv15Szv/FPAS.jpg)

The first part is Feature Pyramid Aggregation Structure (FPAS). In Figure, FPAS are to acquire and fuse attention features from different levels from top-down and bottom-up. This structure is inspired by the approach of Path Aggregation Feature Pyramid Network (PAFPN) to enhance in-network feature hierarchy, where top-down paths, lateral connections, and bottom-up paths are augmented to propagate different semantic meanings. Then, PAFPN increases the performance on object detection and instance segmentation tasks. 

## Modified ASPP

![Modified ASPP](https://i.ibb.co/5GfBWsw/Modified-ASPP-Structure.jpg)

A modified Atrous Spatial Pyramid Pooling (ASPP) module after FPAS is shown in the first figure. It is used for channel attention purpose, which is detailed in the above figure. This part acquires the output of FPAS to process via Atrous convolutions, and the ECA (Effective Channel Attention) modules. 

## Multi-binned classification and regression

![Multi-binned Classification and Regression](https://i.ibb.co/Mk3jQdZ/bin-classification-and-regression.jpg)

The last parts are multi-binned classification regression heads are used to convert the aggregated image features (4x4x256) into Euler angles. For large model size prevention in FC implementation (Fully Connected), a 1x1 GAP layer is applied first  before entering 6 sets of FC layers.

The three sets of FC layers are 10-bin angle classification which is the class of the angle lie in the array [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80], and another three sets of FC layers are 20-bin shifted angle classification which is the addition of the angle from the result of 10-bin angle classification. The addition value starts from 0 to 19. 

---

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

---

## How to train

Train by using command below:

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epoch <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size>
```

If you want to make a transfer learning, you have to type like this below.

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epoch <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size> --transfer 1 --snapshot <snapshot path>
```

If you want to continue training, you have to type like this below.

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epoch <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size> --transfer 0 --snapshot <snapshot path>
```

#### DAD-3DHeads

Train on DAD first. If you do not want to train, you can download the [pretrained model](https://drive.google.com/file/d/1BiDf62gjkffXQYg2HgLisMsfldEXX49P/view?usp=sharing).

```
python train.py --dataset DAD --data_dir datasets/DAD/ --filename_list datasets/DAD/train.json --num_epoch 50 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string DAD --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt      
```

#### 300W_LP

Train on 300W_LP.

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt               
```

The recommended approach is to transfer the pretrained model from DAD-3DHeads.

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt --transfer 1 --snapshot <pretrained DAD-3DHeads path>
```

#### BIWI

Train on BIWI.

```
python train.py --dataset BIWI --data_dir datasets/BIWI/ --filename_list datasets/BIWI/biwi_train_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_test_list.txt     
```

The recommended approach is to transfer the pretrained model from 300W_LP which is fine-tune for BIWI.

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epoch 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt --transfer 1 --snapshot <pretrained 300W_LP path>
```

---

## How to evaluate

Test on the desired datasets. How to do.

#### For train by 300W_LP

Test on AFLW2000 on the whole snapshot folder

```
python test.py --dataset AFLW2000 --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 224 --crop 0
```

Test on AFLW2000 on the specific snapshot file.

```
python test.py --dataset AFLW2000 --val_dataset AFLW2000 --val_data_dir datasets/AFLW2000 --val_filename_list datasets/AFLW2000/aflw2000_list.txt --snapshot <snapshot path> --input_size 224 --crop 0
```


Test on BIWI on the whole snapshot folder

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 240 --crop_size 224 --crop 1
```

Test on BIWI on the specific snapshot file.

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_list.txt --snapshot <snapshot path> --input_size 240 --crop_size 224 --crop 1
```

#### For train by BIWI

Test on BIWI

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_test_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --input_size 224 --crop_size 224 --crop 0
```

Test on BIWI

```
python test.py --dataset BIWI --val_dataset BIWI --val_data_dir datasets/BIWI --val_filename_list datasets/BIWI/biwi_test_list.txt --snapshot <snapshot path> --input_size 224 --crop_size 224 --crop 0
```

---

## The result

To evaluate to be similar to the result, train by DAD-3DHeads first. Then, apply the pretrained model on DAD-3DHeads to be trained by 300W_LP.

The pretrained is avilable from Google Drive. The links are provided in the table.

| Angle      | Yaw         | Pitch       | Roll        | Mean        | Link        |
| ---------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| AFLW2000   | 2.84        | 4.11        | 3.00        | 3.42        | [Download](https://drive.google.com/file/d/1Bw0A0PLcvcmkpfC0H_zhEFJ6ukbX4Mm2/view?usp=sharing)    |
| BIWI       | 4.09        | 3.82        | 2.79        | 3.57        | [Download](https://drive.google.com/file/d/1wt5Qn-87-PBXNs5gIuqQ-XEqPECWTm8s/view?usp=sharing)    |
