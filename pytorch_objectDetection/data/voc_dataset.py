import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
import os
import cv2
import torchvision
import json
import logging
import sys
import re
import xml.etree.ElementTree as ET

from data.transforms import (
    get_test_transforms,
    get_training_transforms,
)


def collate_fn(batch):
    return tuple(zip(*batch))

# Class names.
CLASSES= [
    '__background__',
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC= 21

def getDatasetPath(fileName):
    return os.path.join(os.path.dirname(__file__), f"VOC/{fileName}")

class VOCDataModule(pl.LightningDataModule):
    """
    Class describing the VOC Dataset containing a list of images

    Attributes:
        images (List of VOC)  All images of the dataset
        train_data_path (string):           Path of the dataset (.yaml)
        test_data_path (string):           Path of the dataset (.yaml)
    """
    def __init__(
        self,
        classes,
        batch_size=4,
        num_workers=1,
        random_state=None,
    ):
        super().__init__()
        self.images = []
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_set = VOCDataset(
            self.classes,
            get_training_transforms(),
        )
        original_train_dataset = VOCDataset(
            self.classes,
            get_test_transforms(),
        )
        
        self.test_set = VOCDataset(
            self.classes,
            get_test_transforms(),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.valid_set,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         collate_fn=collate_fn,
    #     )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, classes, transforms):
        if isinstance(classes, list):
            self.classes = classes
        else:
            self.classes = [classes]
        self.transforms = transforms
        self.imagePaths = []

        logging.info("Loading VOC Dataset!")

        for imgClass in self.classes:
            if imgClass not in CLASSES:
                logging.exception(f"Provided class is {imgClass} not labeled in VOC dataset")
                continue

            images = []
            trainDataPath = getDatasetPath(f"ImageSets/Main/{imgClass}_train.txt")
            if os.path.exists(trainDataPath) is not None:
                with open(trainDataPath, "r") as fp:
                    images = self.parseData(fp)
            else:
                logging.exception(
                    "Opening VOC Dataset for class: {} "
                    "failed.".format(self.imgClass)
                )
                sys.exit(1)
                return False
            
            for img in images:
                self.imagePaths.append(img)

    def parseData(self, file):
        logging.info(f"Parsing class text file: {file.name}")
        images = []
        for line in file.readlines():
            line = line.replace('\n','')
            pattern = r'([ ]+)'
            output = re.split(pattern, line)
            images.append([output[0], output[2]])

        return images

    def __getitem__(self, idx):
        image_data = self.imagePaths[idx]
        image_path = getDatasetPath(f"JPEGImages/{image_data[0]}.jpg")
        annotation_path = getDatasetPath(f"Annotations/{image_data[0]}.xml")
        image_name = os.path.basename(image_path)

        # Load image from file path, do debayering and shift
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if os.path.splitext(image_path)[1] == ".tiff":
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2BGR)
                # Images are saved in 12 bit raw -> shift 4 bits
                image = np.right_shift(image, 4)
                image = image.astype(np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32")
            image /= 255

        else:
            logging.error("Image {} not found. Please check image file paths!".format(image_path))
            sys.exit(1)
            return False, np.array()
        
        tree = ET.parse(annotation_path)

        boxes = []
        labels = []
        for row in tree.findall('object'):
            name = row.find('name').text
            bndbox = row.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(image_data[1]))
            # labels.append(name)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:
            augmented = self.transforms(
                image=image, bboxes=target["boxes"], labels=labels
            )
            image = augmented["image"]
            target["boxes"] = torch.as_tensor(augmented["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(augmented["labels"])

        return image, target, image_name

    def __len__(self):
        return len(self.imagePaths)