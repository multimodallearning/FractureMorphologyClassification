import h5py
import torch
import numpy as np
import os
import re
from imageio import v3 as iio
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from utils import normalize_code, insert_bones_to_code, bbox_matches_line, get_bboxes
from utils import map_label_to_class, get_laterality, get_projection, get_roi_from_bbox, get_label, mapping

yolo = False # True whether the bounding box labels in labelpath are yolo generated ones, else False for GT bboxes
data_folder = "fracturemorphologyclassification/data/preprocessed_images"
labelpath = "fracturemorphologyclassification/data/gt_bboxes"
dataset = pd.read_csv("fracturemorphologyclassification/data/dataset.csv")
output_path = "fracturemorphologyclassification/data/output"
segmentations_path = "fracturemorphologyclassification/data/raw_segmentations_all.h5"
os.makedirs(output_path, exist_ok=True)

for file in tqdm(sorted(os.listdir(labelpath))):
    filename = file[:-4]
    if filename.startswith("."):
        continue

    bboxes = get_bboxes(filename, labelpath)
    if len(bboxes) == 0:
        #No bounding boxes, keep files as it was
        with open(os.path.join(output_path, file), "w") as f:
            f.write(open(os.path.join(labelpath, file), "r").read())
        continue

    ao_code = str(get_label(filename, dataset))
    if ao_code == "nan":
        # No AO codes are given for this file so the label file isn't changed
        with open(os.path.join(output_path, file), "w") as f:
            f.write(open(os.path.join(labelpath, file), "r").read())
        continue

    ao_code = re.split(r'[;,]', ao_code)
    ao_code = normalize_code(ao_code)

    data = []

    # Case when one bounding box and one AO code are given for an image -> direct assignment of the bounding box to the AO code
    if len(bboxes) == 1 and len(ao_code) == 1:

        box = bboxes[0] # there is only one bounding box
        
        if len(box) == 0:
            with open(os.path.join(output_path, file), "w") as f:
                f.write(open(os.path.join(labelpath, file), "r").read())
            continue
       
        with open(os.path.join(labelpath, file), "r") as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if int(parts[0]) == 3:
                parts.append(ao_code[0].strip())
            modified_lines.append(" ".join(parts))

        with open(os.path.join(output_path, file), "w") as f:
            for line in modified_lines:
                f.write(line + "\n")
            
    # Case when there is more than one bounding box or one AO code for an image -> assignment of the bounding boxes to the AO codes is needed
    elif len(bboxes) > 1 or len(ao_code) > 1:
        
        # Check if there are segmentations since the segmentations where only created for projections 1
        # so the cases which projection 2 can't be used here
        if get_projection(filename, dataset) == 2:
            with open(os.path.join(output_path, file), "w") as f:
                f.write(open(os.path.join(labelpath, file), "r").read())
            continue

        new_ao_code = []
        for code in ao_code:
            # Extent AO codes which include both bone fractures
            if code in mapping["Both bones"]:
                new_ao_code.extend(insert_bones_to_code(code))
            else:
                new_ao_code.append(code)
        ao_code = new_ao_code

        # Read image
        img = iio.imread(os.path.join(data_folder, filename + ".png"))
        img_height, img_width = img.shape[:2]
        
        bbox_classes = []

        for box in bboxes: 
            # Load segmentations for the four bone classes
            h5_file = h5py.File(segmentations_path, 'r')
            ds_saved_seg = h5_file['segmentation_mask']
            try:
                seg_masks = torch.from_numpy(ds_saved_seg[filename][[0, 1, 15, 16], :])
            except KeyError:
                with open(os.path.join(output_path, file), "w") as f:
                    f.write(open(os.path.join(labelpath, file), "r").read())
                continue
            # Interpolate the segmentations to the image size
            seg_masks = F.interpolate(seg_masks.float().unsqueeze(0), size=img.shape, mode='nearest').squeeze(0).bool()
            if get_laterality(filename, dataset) == "R":
                seg_masks = torch.flip(seg_masks, [2])
            
            cropped_masks = get_roi_from_bbox(box, yolo, img_width, img_height, seg_masks)
            # Determine the overlap scores of each bone class in the roi of the bounding box
            overlap_scores = cropped_masks.sum(dim=(1, 2))

            # Determine bone class which highest overlap score
            best_idx = int(np.argmax(overlap_scores))
            label_names = ["Epiphyse Radius", "Epiphyse Ulna", "Radius", "Ulna"]

            bbox_classes.append((box, label_names[best_idx]))


        label_classes = [map_label_to_class(label, mapping) for label in ao_code]
        # Check if all labels can be assigned, else non of them is used
        if len(set(label_classes)) != len(label_classes):
            with open(os.path.join(output_path, file), "w") as f:
                f.write(open(os.path.join(labelpath, file), "r").read())
            continue

        # Apply assignment of label classes to bounding box classes
        assignments = []
        for i in range(len(bbox_classes)):
            for j in range(len(label_classes)):
                if bbox_classes[i][1] == label_classes[j]:
                    assignments.append((bbox_classes[i][0], ao_code[j]))  # bbox i bekommt label j
        # If no assigment can be applied, then the file is saved as it was
        if len(assignments) == 0:
            with open(os.path.join(output_path, file), "w") as f:
                f.write(open(os.path.join(labelpath, file), "r").read())
            continue
        # Check if a bounding box or an AO code were used for multiple assignments, if yes there are not used,
        # cause an unambiguous assignment is needed
        if len({tuple(a[0]) for a in assignments}) != len(assignments) or len({a[1] for a in assignments}) != len(assignments):

            with open(os.path.join(output_path, file), "w") as f:
                f.write(open(os.path.join(labelpath, file), "r").read())
            continue


        # When there is a clear assignment, it is used to adapt the label file with the corresponding AO code for 
        # the correct bounding box
        with open(os.path.join(labelpath, file), "r") as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if int(parts[0]) == 3:
                #find the corresponding bbox_idx
                for bbox, code in assignments:
                    if bbox_matches_line(bbox, parts):
                        parts.append(code)
                        break
            modified_lines.append(" ".join(parts))

        with open(os.path.join(output_path, file), "w") as f:
            for line in modified_lines:
                f.write(line + "\n")
        
    else:
        with open(os.path.join(output_path, file), "w") as f:
            f.write(open(os.path.join(labelpath, file), "r").read())
        continue

                