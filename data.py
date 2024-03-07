# The code is based on Ruiz's HopeNet.
# This is from the repository.
# https://github.com/natanielruiz/deep-head-pose
import os, torch, cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import utils

def get_list_from_filenames(file_path):
    # input:    relative path to file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

# AFLW2000 Dataset
class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, albu_transform, transform, image_mode='RGB', increase_gradient = False, sixd = True, ad = 0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.transform_albu = albu_transform

        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(self.X_train)
        self.increase_gradient = increase_gradient
        self.sixd = sixd

        self.img_ext = '.jpg'
        self.annot_ext = '.mat'
        self.ad = ad

    def __len__(self):
        # 2,000 images
        return self.length

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext)) 
        img = img.convert(self.image_mode) 

        raw_img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext)) #np.array(img)

        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt3d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = self.ad
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        pose = utils.get_ypr_from_mat(mat_path)

        # And convert to degrees.
        pose = np.array(pose)
        pitch, yaw, roll = np.rad2deg(pose)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, raw_img, index
    
# BIWI Dataset
class BIWI_kinect(Dataset):
    def __init__(self, root_path, filename_list, albu_transform, torch_transform, mode = 'train', img_ext='.jpg', annot_ext='.npz', image_mode='RGB', increase_gradient = False, sixd = True):
        self.root_path = root_path
        self.transform_albu = albu_transform
        self.transform = torch_transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.X_train, self.y_train = [], []

        self.X_train = get_list_from_filenames(filename_list)
        self.y_train = get_list_from_filenames(filename_list)

        self.image_mode = image_mode
        self.length = len(self.X_train)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root_path, self.X_train[index] + ".npz"))

        raw_img = np.uint8(data['image'].copy())
        img = np.uint8(data['image'])
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.convert(self.image_mode)
        
        yaw, pitch, roll = data['pose']
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        
        if self.transform is not None:
            img = self.transform(img)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = torch.LongTensor(binned_pose)

        return img, labels, cont_labels, raw_img, index

    def __len__(self):
        # 15,667
        return self.length
