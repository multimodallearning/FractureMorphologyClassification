import os
import torch
import numpy as np
from torchmetrics import ConfusionMatrix
import re

"""
Mapping of the AO codes to their corresponding morphology class
"""
MORPHOLOGY_AO = {
    "Transverse":("22-D/4.1", "22r-D/4.1", "22u-D/4.1", "23-M/3.1", "23r-M/3.1", "23u-M/3.1"),
    "Greenstick":("22-D/2.1", "22r-D/2.1", "22u-D/2.1"),
    "Torus":("23-M/2.1", "23r-M/2.1", "23u-M/2.1"),
    "SalterII":("23-E/2.1", "23r-E/2.1", "23u-E/2.1"),
    "Avulsion":("23-E/7", "23r-E/7", "23u-E/7")
}

"""
Mapping of the AO codes to their corresponding morphology class including the healthy class for FP-reduction
"""
MORPHOLOGY_AO_FPREDUCTION = {
    "Transverse":("22-D/4.1", "22r-D/4.1", "22u-D/4.1", "23-M/3.1", "23r-M/3.1", "23u-M/3.1"),
    "Greenstick":("22-D/2.1", "22r-D/2.1", "22u-D/2.1"),
    "Torus":("23-M/2.1", "23r-M/2.1", "23u-M/2.1"),
    "SalterII":("23-E/2.1", "23r-E/2.1", "23u-E/2.1"),
    "Avulsion":("23-E/7", "23r-E/7", "23u-E/7"),
    "Healthy":("nofracture",)
}


def get_label(filestem, label_file):
    """Returns the AO code for a given file

    Args:
        filestem (str): File stem for which the AO code should be determined
        label_file (pd.Dataframe): Label file

    Returns:
        _type_: _description_
    """
    label = label_file.loc[label_file['filestem'] == filestem, 'ao_classification'].values
    if len(label) == 0:
        return "nan"
    return label[0]

def get_morphology_one_hot_tensor(ao_code_list):
    """Returns an one hot encoded vector from an list of AO codes

    Args:
        ao_code_list (list): List of AO codes

    Returns:
        torch.tensor: One hot encoded vector of AO codes in an image
    """
    class_names = list(MORPHOLOGY_AO.keys())
    one_hot = torch.zeros(len(class_names), dtype=torch.float32)

    for code in ao_code_list:
        for idx, morph in enumerate(class_names):
            if code in MORPHOLOGY_AO[morph]:
                one_hot[idx] = 1

    return one_hot

def normalize_code(code_list):
    """Normalizes a list of AO codes

    Args:
        code_list (list): List of initial AO codes which should be normalized

    Returns:
        list: List of the normalized AO codes
    """
    # Fix missing slash between letter and number
    adapted_codes = []
    for code in code_list:
        code = code.replace(" ", "")
        code = re.sub(r'(?<=[A-Z])(?=\d)', '/', code)
        
        # Fix slash between prefix and main part
        code = re.sub(r'(\d{2,}u)/([A-Z])', r'\1-\2', code)
        
        # Fix missing number at the end (assuming closest match)
        if code.endswith('.'):
            for category, codes in MORPHOLOGY_AO.items():
                for valid_code in codes:
                    if valid_code.startswith(code[:-1]):  # Check if all but the last digit match
                        adapted_codes.append(valid_code)  # Use the closest match
        elif code.endswith('/1') or code.endswith('/4'):
            code += ".1"
            adapted_codes.append(code)
        elif code == "23r-D/2.1":
            adapted_codes.append("22r-D/2.2")
        else:
            adapted_codes.append(code)
    
    return adapted_codes

def get_ground_truth_bboxes(filestem, gt_bbox_path):
    """Return ground truth bounding boxes and the corresponding ao codes

    Args:
        filestem (str): File for which the gt boxes should be determined
        gt_bbox_path (str): Path to the ground truth files

    Returns:
        (list, list): Bounding boxes and the corresponding AO codes
    """
    bboxes = []
    gt_codes = []
    filepath = os.path.join(gt_bbox_path, filestem + ".txt")
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) == 3 and len(parts)==6:  #code three for fracture and existing AO code
                    bboxes.append(map(float, parts[1:-1]))
                    gt_codes.append(parts[-1])
    return bboxes, gt_codes

def get_bbox_coordinates(bbox, img_width, img_height):
    """Return the bounding box coordinates a given bounding box

    Args:
        bbox (tuple): Bounding box
        img_width (int): Image width
        img_height (int): Image height

    Returns:
        tuple: corner coordinates of the bounding box
    """
    x, y, w, h = bbox
    x *= img_width
    y *= img_height
    w *= img_width
    h *= img_height

    # Convert to top-left corner coordinates
    x1 = x - w / 2
    y1 = y - h / 2

    return (x1, y1, x1 + w, y1 + h)

def get_intersection_over_union(coords1, coords2):
    """Determines the IoU from two coordinates tuples of two bounding boxes

    Args:
        coords1 (tuple): First bounding box coordinates
        coords2 (_type_): Second bounding box coordinates

    Returns:
        float: IoU of the bounding boxes
    """
    
    A = max(coords1[0], coords2[0])
    B = max(coords1[1], coords2[1])
    C = min(coords1[2], coords2[2])
    D = min(coords1[3], coords2[3])

    inter_area = max(0, C - A) * max(0, D - B)

    X_area = (coords1[2] - coords1[0]) * (coords1[3] - coords1[1])
    Y_area = (coords2[2] - coords2[0]) * (coords2[3] - coords2[1])

    union_area = X_area + Y_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

"""_summary_
Adapted Metrics including false negativs from Yolo detector
"""
class MultiLabelMetricFNAdaption:
    def __init__(self, y_true, y_pred, num_classes, labels, extra_fn=None):
        """Init of the Metric adaption

        Args:
            y_true (torch.tensor): Labels
            y_pred (torch.tensor): Predictions
            num_classes (int): number of classes
            labels (torch.tensor): labels for determing the device
            extra_fn (torch.tensor, optional): Additional false negativs which should be included in the metrics. Defaults to None.
        """
        self.y_true = y_true.clone().detach().long()
        self.y_pred = y_pred.clone().detach().long()
        self.num_classes = num_classes
        self.extra_fn = extra_fn.clone().detach().long() if extra_fn is not None else torch.zeros(num_classes)

        self.cm = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(labels.device)(y_pred, y_true).cpu().numpy()

        # TP, TN, FP, FN berechnen
        self.tp, self.fp, self.fn, self.tn = self._compute_counts()

        # Zusätzliche FN addieren
        self.fn = (torch.tensor(self.fn) + self.extra_fn).tolist()

    def _compute_counts(self):
        """Compute TruePositive, FalsePositive, FalseNegative, TrueNegative

        Returns:
            tuple: TP, FP, FN, TN
        """
        TP = np.diag(self.cm).astype(int)
        FP = np.sum(self.cm, axis=0) - TP
        FN = np.sum(self.cm, axis=1) - TP
        TN = np.sum(self.cm) - (TP + FP + FN)
        return TP, FP, FN, TN
    
    def compute(self, metric="accuracy"):
        """Computes the given metric per class and averaged

        Args:
            metric (str, optional): Metric which should be determined. Defaults to "accuracy".

        Raises:
            ValueError: When an unknown metric should be determined, only accuracy, F1-score, precision and recall are implemented.

        Returns:
            dict: Metrics per class and macro averaged metric
        """
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        if metric == "accuracy":
            per_class = (tp + tn) / (tp + fp + fn + tn + 1e-9)
        elif metric == "f1":
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            per_class = (2 * precision * recall) / (precision + recall + 1e-9)
        elif metric == "precision":
            per_class = tp / (tp + fp + 1e-9)
        elif metric == "recall":
            per_class = tp / (tp + fn + 1e-9)
        else:
            raise ValueError("Unknown metric")
        
        return {
            "per_class": per_class.tolist(),
            "macro_avg": np.mean(per_class).item()
        }