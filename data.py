import json
import random
import os
import re
from collections import defaultdict

import cv2
import torchvision
from skimage.transform import downscale_local_mean
from skimage import io
import numpy as np
import torch
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader


AU_CROP = {
    1: "upper",
    2: "upper",
    4: "upper",
    6: "upper",
    7: "upper",
    10: "lower",
    12: "lower",
    14: "lower",
    15: "lower",
    17: "lower",
    23: "lower",
    24: "lower"
}


def get_frames(coding_file, AUs):
    data = genfromtxt(coding_file, dtype=int, delimiter=',')
    data = data[1:, [0] + AUs]
    labels = data[:, 1:]
    labels[labels == 9] = 0
    return data


def get_sample_indices(num_samples, data_split, seed):
    # Pick indices of females and males together
    # Females index range from 1 to 23 (as 1-23 in code)
    # Males index from 1 to 18 (as 24-41)
    random.seed(seed)

    samples = random.choices(range(1, 42), k=num_samples)

    train_split = int(num_samples * data_split[0])
    val_split = int(num_samples * (data_split[0] + data_split[1]))

    def sort_by_gender(sample_list):
        sorted_list = defaultdict(list)
        for s in sample_list:
            if s <= 23:
                sorted_list["F"].append(s)
            else:
                sorted_list["M"].append(s-23)
        return sorted_list

    train_indices = sort_by_gender(samples[:train_split])
    val_indices = sort_by_gender(samples[train_split:val_split])
    test_indices = sort_by_gender(samples[val_split:])

    return train_indices, val_indices, test_indices


class ImageDataset(Dataset):
    def __init__(self, img_files, labels, crop=None):
        self.img_files = img_files
        self.labels = labels
        self.crop = crop

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.img_files[idx]
        image = io.imread(file)

        if self.crop:
            match = re.match(".*([FM]\d*)", file)
            person = match.group(1)

            with open("face_crop.json") as json_file:
                data = json.load(json_file)
                upper_edge = data[person]["upper"]
                lower_edge = data[person]["lower"]

            if self.crop == "upper":
                image = image[:upper_edge]
                image = cv2.resize(image, (64, 32))
            else:
                image = image[lower_edge:]
                image = cv2.resize(image, (64, 40))
        else:
            image = cv2.resize(image, (64, 64))

        image = torchvision.transforms.functional.to_tensor(image).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.FloatTensor)

        return image, label


class DataGenerator:
    def __init__(
            self,
            coding_folder="data\\AUCoding\\AU_OCC",
            image_folder="data\\images",
            samples=5,
            data_split=(0.6, 0.2, 0.2),
            AUs=[1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24],
            seed=42
    ):
        self.coding_folder = coding_folder
        self.image_folder = image_folder
        self.AUs = AUs

        self.single_AU = len(AUs) == 1

        train_id, val_id, test_id = get_sample_indices(samples, data_split, seed)
        self.train = DataLoader(
            dataset=self.__get_data__(train_id),
            batch_size=64,
            shuffle=True,
            num_workers=0
        )
        self.val = DataLoader(
            dataset=self.__get_data__(val_id),
            batch_size=64,
            num_workers=0
        )
        self.test = DataLoader(
            dataset=self.__get_data__(test_id),
            batch_size=64,
            num_workers=0
        )

    def __get_data__(self, indices):
        image_list = []
        labels_list = []
        for gender, data in indices.items():
            for index in data:
                for task in range(1, 9):
                    header = f'{gender}{index:03d}_T{task}'
                    file = header + ".csv"
                    file_path = os.path.join(self.coding_folder, file)
                    data = get_frames(file_path, self.AUs)
                    frames = data[:, 0]
                    labels = list(data[:, 1:])
                    images = [
                        os.path.join(self.image_folder, f"{header}_{frame:04d}.jpg")
                        for frame in frames
                    ]
                    image_list += images
                    labels_list += labels

        if self.single_AU:
            crop = AU_CROP[self.AUs[0]]
        else:
            crop = None

        return ImageDataset(image_list, labels_list, crop)
