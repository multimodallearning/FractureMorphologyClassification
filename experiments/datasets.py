from utils import get_ground_truth_bboxes, get_bbox_coordinates, get_intersection_over_union, MORPHOLOGY_AO_FPREDUCTION
from utils import get_label, normalize_code, get_morphology_one_hot_tensor, MORPHOLOGY_AO
import imageio.v3 as iio
from torch.utils.data import Dataset
import os
import torch
import kornia.augmentation as K
import torch.nn as nn
import pandas as pd
import re

"""
Dataset for Binary fracture classification on yolo bounding boxes
"""
class GrazerFracturesImageDataset_BinaryFractureClassification(Dataset):
    def __init__(self, data_path, label_path, gt_bbox_path, confidence_threshold, iou_threshold, transform=None):
        """Init function for the dataset

        Args:
            data_path (str): Path to the preprocessed data
            label_path (str): Path to the labels of the yolo
            gt_bbox_path (str): Path to the labels of the GT
            confidence_threshold (float): Confidence score for the yolo detector
            iou_threshold (float): IoU if a patch is a fracture or not
            transform (transforms, optional): Transformations which should be applied to the images. Defaults to None.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.gt_path = gt_bbox_path

        self.data = sorted(os.listdir(data_path))
        self.images = []
        self.labels = []
        self.files = []

        self.fn_labels = []

        for file in self.data:
            filename = file[:-4]

            img = iio.imread(os.path.join(self.data_path, file))
            img_height, img_width = img.shape[:2]

            if not os.path.exists(os.path.join(label_path, filename + ".txt")):
                continue
            with open(os.path.join(label_path, filename + ".txt"), "r") as f:
                lines = f.readlines()
            
            gt_bboxes, gt_codes = get_ground_truth_bboxes(filename, self.gt_path)
            gt_coords = [get_bbox_coordinates(gt_box, img_width, img_height) for gt_box in gt_bboxes]
            matched_gt = set()
            
            for line in lines:
                parts = line.strip().split()
                if int(parts[0]) == 3 and len(parts)==7: # fracture with label
                    if float(parts[5]) < confidence_threshold:
                        continue
                    x, y, w, h = map(float, parts[1:-2])
                    x *= img_width
                    y *= img_height
                    w *= img_width
                    h *= img_height

                    # Convert to top-left corner coordinates
                    x1 = x - w / 2
                    y1 = y - h / 2
                
                    #crop image in the bounding box and save it as png to the folder
                    cropped_img = img[int(y1):int(y1+h), int(x1):int(x1+w)]

                    bbox_coords = get_bbox_coordinates(bbox=map(float, parts[1:-2]), img_width=img_width, img_height=img_height)
                    iou_scores = [get_intersection_over_union(bbox_coords, gt_coord) for gt_coord in gt_coords]
                    if len(iou_scores) == 0 or max(iou_scores) < iou_threshold:
                        ao_code = "nofracture"
                    else:
                        best_idx = torch.argmax(torch.tensor(iou_scores))
                        ao_code = gt_codes[best_idx]  
                        matched_gt.add(int(best_idx))      
                    #find class of MORPHOLOGY_AO that contains the ao_code
                    label = None
                    for key, value in MORPHOLOGY_AO_FPREDUCTION.items():
                        if ao_code in value:
                            label = key
                            break
                    if label is None:
                        continue

                    #append the index of the label to self.labels
                    self.labels.append(0 if list(MORPHOLOGY_AO_FPREDUCTION.keys()).index(label) in [0,1,2,3,4] else 1)
                    self.images.append(cropped_img)
                    self.files.append(filename)
            if label is None:
                continue

            for gt_idx, ao_code in enumerate(gt_codes):
                if gt_idx not in matched_gt:
                    # save class for false negative calculation
                    for key, value in MORPHOLOGY_AO_FPREDUCTION.items():
                        if ao_code in value:
                            self.fn_labels.append(0 if list(MORPHOLOGY_AO_FPREDUCTION.keys()).index(label) in [0,1,2,3,4] else 1)
                            break


        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=25, scale=(0.8, 1.2), translate=(0.1, 0.1))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        image = image.repeat(3, 1, 1)#repeat the image 3 times to get 3 channels
        image = self.augmentations(image)
        image = image.squeeze(0)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

"""
Dataset for multi-label classification on the full-image
"""
class GrazerFracturesImageDataset_MultiLabelWholeImage(Dataset):
    def __init__(self, path_to_images, path_to_labels, transform=None):
        """Init function for the dataset

        Args:
            path_to_imahes (str): Path to the preprocessed data
            path_to_labels (str): Path to the labels of the GT
            transform (transforms, optional): Transformations which should be applied to the images. Defaults to None.
        """
        self.path_to_images = path_to_images
        self.image_paths = sorted([os.path.join(path_to_images, f) for f in os.listdir(path_to_images) if f.endswith('.png')])
        self.transform = transform

        self.label_path = path_to_labels
        self.label_file = pd.read_csv(self.label_path)
        self.ao_codes = self.label_file[["filestem", "ao_classification"]]
        #run trough all images, save them in self.images and then extract index of ao code like its in getitem to safe al labels in self.labels
        self.labels = []
        self.images = []

        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            ao_code = str(get_label(img_name[:-4], self.label_file))
            
            if ao_code == "nan":
                continue
            
            ao_code = re.split(r'[;,]', ao_code)
            ao_code = normalize_code(ao_code)
            one_hot_vector = get_morphology_one_hot_tensor(ao_code)
            

            #append the index of the label to self.labels
            self.labels.append(one_hot_vector)
            self.images.append(iio.imread(img_path))

        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=25, scale=(0.8, 1.2), translate=(0.1, 0.1))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # Read image
        image = self.images[idx]
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        
        #repeat the image 3 times to get 3 channels
        image = image.repeat(3, 1, 1)

        image = self.augmentations(image)

        image = image.squeeze(0)
        
        # Get label
        label = self.labels[idx]
        return image, label.clone().detach().float()

"""
Dataset for multi-class classification on GT bounding boxes
"""
class GrazerFracturesImageDataset_MultiClassGT(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        """Init function for the dataset

        Args:
            data_path (str): Path to the preprocessed data
            label_path (str): Path to the labels of the GT
            transform (transforms, optional): Transformations which should be applied to the images. Defaults to None.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform

        self.data = sorted(os.listdir(data_path))
        self.images = []
        self.labels = []
        self.files = []

        self.fn_labels = []

        for file in self.data:
            filename = file[:-4]

            img = iio.imread(os.path.join(self.data_path, file))
            img_height, img_width = img.shape[:2]

            if not os.path.exists(os.path.join(label_path, filename + ".txt")):
                continue
            with open(os.path.join(label_path, filename + ".txt"), "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if int(parts[0]) == 3 and len(parts)==6: # fracture with label
                    x, y, w, h = map(float, parts[1:-1])
                    x *= img_width
                    y *= img_height
                    w *= img_width
                    h *= img_height

                    # Convert to top-left corner coordinates
                    x1 = x - w / 2
                    y1 = y - h / 2
                
                    #crop image in the bounding box and save it as png to the folder
                    cropped_img = img[int(y1):int(y1+h), int(x1):int(x1+w)]
                    ao_code = parts[-1]

                    #find class of MORPHOLOGY_AO that contains the ao_code
                    label = None
                    for key, value in MORPHOLOGY_AO.items():
                        if ao_code in value:
                            label = key
                            break
                    if label is None:
                        continue

                    #append the index of the label to self.labels
                    self.labels.append(list(MORPHOLOGY_AO.keys()).index(label))
                    self.images.append(cropped_img)
                    self.files.append(filename)


        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=25, scale=(0.8, 1.2), translate=(0.1, 0.1))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # Read image
        image = self.images[idx]
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        
        #repeat the image 3 times to get 3 channels
        image = image.repeat(3, 1, 1)

        image = self.augmentations(image)

        image = image.squeeze(0)
        
        # Get label
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

"""
Dataset for multi-class classification on yolo bounding boxes
"""
class GrazerFracturesImageDataset_YOLO(Dataset):
        def __init__(self, data_path, label_path, gt_bbox_path, confidence_threshold, iou_threshold, transform=None):
            """Init function for the dataset

            Args:
                data_path (str): Path to the preprocessed data
                label_path (str): Path to the labels of the yolo
                gt_bbox_path (str): Path to the labels of the GT
                confidence_threshold (float): Confidence score for the yolo detector
                iou_threshold (float): IoU if a patch is a fracture or not
                transform (transforms, optional): Transformations which should be applied to the images. Defaults to None.
            """
            self.data_path = data_path
            self.label_path = label_path
            self.transform = transform
            self.gt_path = gt_bbox_path

            self.data = sorted(os.listdir(data_path))
            self.images = []
            self.labels = []
            self.files = []

            self.fn_labels = []

            for file in self.data:
                filename = file[:-4]

                img = iio.imread(os.path.join(self.data_path, file))
                img_height, img_width = img.shape[:2]

                if not os.path.exists(os.path.join(label_path, filename + ".txt")):
                    continue
                with open(os.path.join(label_path, filename + ".txt"), "r") as f:
                    lines = f.readlines()
                
                gt_bboxes, gt_codes = get_ground_truth_bboxes(filename, self.gt_path)
                gt_coords = [get_bbox_coordinates(gt_box, img_width, img_height) for gt_box in gt_bboxes]
                matched_gt = set()
                
                for line in lines:
                    parts = line.strip().split()
                    if int(parts[0]) == 3 and len(parts)==7: # fracture with label
                        if float(parts[5]) < confidence_threshold:
                            continue
                        x, y, w, h = map(float, parts[1:-2])
                        x *= img_width
                        y *= img_height
                        w *= img_width
                        h *= img_height

                        # Convert to top-left corner coordinates
                        x1 = x - w / 2
                        y1 = y - h / 2
                    
                        #crop image in the bounding box and save it as png to the folder
                        cropped_img = img[int(y1):int(y1+h), int(x1):int(x1+w)]

                        bbox_coords = get_bbox_coordinates(bbox=map(float, parts[1:-2]), img_width=img_width, img_height=img_height)
                        iou_scores = [get_intersection_over_union(bbox_coords, gt_coord) for gt_coord in gt_coords]
                        if len(iou_scores) == 0 or max(iou_scores) < iou_threshold:
                            a = 1
                        else:
                            best_idx = torch.argmax(torch.tensor(iou_scores))
                            a = 1 
                            matched_gt.add(int(best_idx))  
                        ao_code = parts[-1]    
                        #find class of MORPHOLOGY_AO that contains the ao_code
                        label = None
                        for key, value in MORPHOLOGY_AO.items():
                            if ao_code in value:
                                label = key
                                break
                        if label is None:
                            continue

                        #append the index of the label to self.labels
                        self.labels.append(list(MORPHOLOGY_AO.keys()).index(label))
                        self.images.append(cropped_img)
                        self.files.append(filename)

                for gt_idx, ao_code in enumerate(gt_codes):
                    if gt_idx not in matched_gt:
                        # save for false negatives
                        for key, value in MORPHOLOGY_AO.items():
                            if ao_code in value:
                                self.fn_labels.append(list(MORPHOLOGY_AO.keys()).index(key))
                                break


            self.augmentations = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomAffine(degrees=25, scale=(0.8, 1.2), translate=(0.1, 0.1))
            )

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            
            # Read image
            image = self.images[idx]
            
            # Convert to tensor
            if self.transform:
                image = self.transform(image)
            
            #repeat the image 3 times to get 3 channels
            image = image.repeat(3, 1, 1)

            image = self.augmentations(image)

            image = image.squeeze(0)
            
            # Get label
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)

"""
Dataset for multi-class classification on yolo bounding boxes with false positive reduction
"""
class GrazerFracturesImageDataset_YOLOFPREDUCTION(Dataset):
    def __init__(self, data_path, label_path, gt_bbox_path, confidence_threshold, iou_threshold, transform=None):
        """Init function for the dataset

            Args:
                data_path (str): Path to the preprocessed data
                label_path (str): Path to the labels of the yolo
                gt_bbox_path (str): Path to the labels of the GT
                confidence_threshold (float): Confidence score for the yolo detector
                iou_threshold (float): IoU if a patch is a fracture or not
                transform (transforms, optional): Transformations which should be applied to the images. Defaults to None.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.gt_path = gt_bbox_path

        self.data = sorted(os.listdir(data_path))
        self.images = []
        self.labels = []
        self.files = []

        self.fn_labels = []

        for file in self.data:
            filename = file[:-4]

            img = iio.imread(os.path.join(self.data_path, file))
            img_height, img_width = img.shape[:2]

            if not os.path.exists(os.path.join(label_path, filename + ".txt")):
                continue
            with open(os.path.join(label_path, filename + ".txt"), "r") as f:
                lines = f.readlines()
            
            gt_bboxes, gt_codes = get_ground_truth_bboxes(filename, self.gt_path)
            gt_coords = [get_bbox_coordinates(gt_box, img_width, img_height) for gt_box in gt_bboxes]
            matched_gt = set()
            
            for line in lines:
                parts = line.strip().split()
                if int(parts[0]) == 3 and len(parts)==7: # fracture with label
                    if float(parts[5]) < confidence_threshold:
                        continue
                    x, y, w, h = map(float, parts[1:-2])
                    x *= img_width
                    y *= img_height
                    w *= img_width
                    h *= img_height

                    # Convert to top-left corner coordinates
                    x1 = x - w / 2
                    y1 = y - h / 2
                
                    #crop image in the bounding box and save it as png to the folder
                    cropped_img = img[int(y1):int(y1+h), int(x1):int(x1+w)]

                    bbox_coords = get_bbox_coordinates(bbox=map(float, parts[1:-2]), img_width=img_width, img_height=img_height)
                    iou_scores = [get_intersection_over_union(bbox_coords, gt_coord) for gt_coord in gt_coords]
                    if len(iou_scores) == 0 or max(iou_scores) < iou_threshold:
                        ao_code = "nofracture"
                    else:
                        best_idx = torch.argmax(torch.tensor(iou_scores))
                        ao_code = gt_codes[best_idx]  
                        matched_gt.add(int(best_idx))      
                    #find class of MORPHOLOGY_AO that contains the ao_code
                    label = None
                    for key, value in MORPHOLOGY_AO_FPREDUCTION.items():
                        if ao_code in value:
                            label = key
                            break
                    if label is None:
                        continue

                    #append the index of the label to self.labels
                    self.labels.append(list(MORPHOLOGY_AO_FPREDUCTION.keys()).index(label))
                    self.images.append(cropped_img)
                    self.files.append(filename)

            for gt_idx, ao_code in enumerate(gt_codes):
                if gt_idx not in matched_gt:
                    # save for false negatives
                    for key, value in MORPHOLOGY_AO_FPREDUCTION.items():
                        if ao_code in value:
                            self.fn_labels.append(list(MORPHOLOGY_AO_FPREDUCTION.keys()).index(key))
                            break

            self.augmentations = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomAffine(degrees=25, scale=(0.8, 1.2), translate=(0.1, 0.1))
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # Read image
        image = self.images[idx]
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        
        #repeat the image 3 times to get 3 channels
        image = image.repeat(3, 1, 1)

        image = self.augmentations(image)

        image = image.squeeze(0)
        
        # Get label
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)