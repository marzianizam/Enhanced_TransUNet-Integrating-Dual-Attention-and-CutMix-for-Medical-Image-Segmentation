import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy.ndimage import zoom as scipy_zoom
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_zoom(image, label, zoom_range=(1.2, 2.0)):
    h, w = image.shape
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

    # Calculate new dimensions
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

    # Randomly choose the center of the zoomed area to ensure it's within the bounds
    center_x, center_y = np.random.randint(new_w // 2, w - new_w // 2), np.random.randint(new_h // 2, h - new_h // 2)

    # Crop and zoom
    cropped_image = image[center_y - new_h // 2:center_y + new_h // 2, center_x - new_w // 2:center_x + new_w // 2]
    cropped_label = label[center_y - new_h // 2:center_y + new_h // 2, center_x - new_w // 2:center_x + new_w // 2]

    # Resize to original dimensions
    resized_image = ndimage.zoom(cropped_image, (zoom_factor, zoom_factor), order=3)
    resized_label = ndimage.zoom(cropped_label, (zoom_factor, zoom_factor), order=0)

    return resized_image, resized_label



def cutmix(image1, label1, image2, label2, alpha=1.0):
    h, w = image1.shape 

    # Get mixing mask size and location
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # Random location within the original image
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Apply patch
    image1[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
    label1[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]

    # Adjust labels proportionally to the area being mixed
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return image1, label1

class RandomGenerator(object):
    def __init__(self, dataset,output_size):
        self.dataset = dataset
        self.output_size = output_size
        

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        idx2 = np.random.randint(len(self.dataset))
        image2, label2 = self.dataset.get_raw_sample(idx2)
        
        if random.random() > 0.66:  
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.33:  
            image, label = random_rotate(image, label)
        else:    
            image, label = cutmix(image, label, image2, label2)
        #else:
            #image, label = random_zoom(image, label)  # Apply zoom
            

        # Resizing to output size, if necessary
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = ndimage.zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = ndimage.zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()
        sample = {'image': image, 'label': label}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir


    def __len__(self):
        return len(self.sample_list)

    def get_raw_sample(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        return image, label

    def __getitem__(self, idx):
        image, label = self.get_raw_sample(idx)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
