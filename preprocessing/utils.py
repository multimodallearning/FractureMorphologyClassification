
import numpy as np
import re

"""
Mapping of the morphology classes to their corresponding AO codes
"""
MORPHOLOGY_AO = {
    "Transverse":("22-D/4.1", "22r-D/4.1", "22u-D/4.1", "23-M/3.1", "23r-M/3.1", "23u-M/3.1"),
    "Oblique":("22-D/5.1", "22r-D/5.1", "22u-D/5.1"),
    "Bowing":("22-D/1.1", "22r-D/1.1", "22u-D/1.1"),
    "Greenstick":("22-D/2.1", "22r-D/2.1", "22u-D/2.1"),
    "Multifragmentary":("22-D/4.2", "22r-D/4.2", "22u-D/4.2", "22-D/5.2", "22r-D/5.2", "22u-D/5.2", "23-M/3.2", "23r-M/3.2", "23u-M/3.2", "23-E/2.2", "23u-E/2.2", "23r-E/2.2", "23r-E/4.2", "23u-E/4.2"),
    "Torus":("23-M/2.1", "23r-M/2.1", "23u-M/2.1"),
    "SalterI":("23-E/1.1", "23r-E/1.1", "23u-E/1.1"),
    "SalterII":("23-E/2.1", "23r-E/2.1", "23u-E/2.1"),
    "SalterIII":("23-E/3.1", "23r-E/3", "23u-E/3"),
    "SalterIV":("23-E/4.1", "23r-E/4.1", "23u-E/4.1"),
    "Avulsion":("23-E/7", "23r-E/7", "23u-E/7")
}

"""
Mapping of the bone classes to their corresponding AO codes
"""
mapping = {
    "Radius": {
        "22r-D/4.1", "23r-M/3.1", "22r-D/5.1", "22r-D/1.1",
        "22r-D/2.1", "22r-D/4.2", "22r-D/5.2", "23r-M/3.2", "23r-M/2.1"
    },
    "Ulna": {
        "22u-D/4.1", "23u-M/3.1", "22u-D/5.1", "22u-D/1.1",
        "22u-D/2.1", "22u-D/4.2", "22u-D/5.2", "23u-M/3.2", "23u-M/2.1"
    },
    "Epiphyse Radius": {
        "23r-E/1.1", "23r-E/2.1", "23r-E/2.2", "23r-E/3.1",
        "23r-E/4.1", "23r-E/4.2", "23r-E/7"
    },
    "Epiphyse Ulna": {
        "23u-E/1.1", "23u-E/2.1", "23u-E/2.2", "23u-E/3.1",
        "23u-E/4.1", "23u-E/4.2", "23u-E/7"
    },
    "Both bones": {
        "22-D/4.1", "23-M/3.1", "22-D/5.1", "22-D/1.1", "22-D/2.1",
        "22-D/4.2", "22-D/5.2", "23-M/3.2", "23-M/2.1",
        "23-E/1.1", "23-E/2.1", "23-E/2.2", "23-E/3.1",
        "23-E/4.1", "23-E/7"
    }
}

def normalize_code(code_list):
    """Normalizes a list of AO codes

    Args:
        code_list (list): List of initial AO codes which should be normalized

    Returns:
        list: List of the normalized AO codes
    """
    
    adapted_codes = []
    for code in code_list:
        code = code.replace(" ", "")
        # Fix missing slash between letter and number
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
        # Fix non existent AO code of the given dataset
        elif code == "23r-D/2.1":
            adapted_codes.append("22r-D/2.2")
        else:
            # Add normalized code to the resulting codes
            adapted_codes.append(code)
    
    return adapted_codes

def insert_bones_to_code(code):
    """Inserts into a given AO code for both bones the radius and ulna chars and return both codes

    Args:
        code (str): AO code

    Returns:
        list: List with two string, showing both the given AO code with the radius and ulna version
    """
    return [code[:2] + "r" + code[2:], code[:2] + "u" + code[2:]]

def bbox_matches_line(bbox, parts):
    """Check of a given bounding box matches a given line of a label file

    Args:
        bbox (array_like): Bounding box
        parts (list): List of the elements of a single line of a label file

    Returns:
        bool: Returns True if the bounding box matches the line of the file else False
    """
    try:
        line_values = np.array([float(x) for x in parts])
        return np.allclose(bbox, line_values)
    except ValueError:
        return False
    
def get_bboxes(filestem, labelpath):
    """Returns the bounding boxes of all fractures of a given filestem

    Args:
        filestem (str): Filestem of which the bounding boxes should be determined
        labelpath (str): Path where the label files are suited

    Returns:
        list: All fracture bounding boxes of the given filestem
    """
    bbox = np.loadtxt(labelpath + "/" + filestem + ".txt")
    if bbox.ndim == 1:
        bbox = bbox.reshape(1, -1)
    bboxes = []
    for box in bbox:
        if len(box) == 0:
            continue
        #Check if the bounding box shows a fracture which has the code 3
        if box[0] == 3:
            bboxes.append(box)
    return bboxes

def map_label_to_class(label, mapping):
    """Returns the bone class of a given AO code

    Args:
        label (str): AO code
        mapping (dict): Mapping of bone classes to their corresponding AO codes

    Returns:
        str: The corresponding bone class of the given AO code
    """
    for cls, values in mapping.items():
        if label.strip() in values:
            return cls
    return None  

def get_laterality(filestem, dataset):
    """Returns the laterity of a given image based on its filestem

    Args:
        filestem (str): Filestem of the image
        dataset (pd.Dataframe): Dataframe of the dataset containing the metadata

    Returns:
        str: Returns the laterity of the image: L or R (left or right)
    """
    laterality = dataset.loc[dataset['filestem'] == filestem, 'laterality'].values
    return laterality[0]

def get_projection(filestem, dataset):
    """Returns the projection of a given image based on its filestem

    Args:
        filestem (str): Filestem of the image
        dataset (pd.Dataframe): Dataframe of the dataset containing the metadata

    Returns:
        str: Returns the projection of the image: 1 or 2
    """
    projection = dataset.loc[dataset['filestem'] == filestem, 'projection'].values
    return projection[0]

def get_label(filestem, dataset):
    """Returns the AO code(s) of a given image based on its filestem

    Args:
        filestem (str): Filestem of the image
        dataset (pd.Dataframe): Dataframe of the dataset containing the metadata

    Returns:
        str: Returns the AO code(s)
    """
    label = dataset.loc[dataset['filestem'] == filestem, 'ao_classification'].values
    return label[0]

def get_roi_from_bbox(box, yolo, img_width, img_height, seg_masks):
    """Returns the region of interest of the segmentations a given image in the given bounding box area

    Args:
        box (array_like): The bounding box
        yolo (bool): If the bounding box was generated by the yolo or not
        img_width (int): Image width
        img_height (int): Image Height
        seg_masks (array_like): Segmentations masks for the bone classes for the whole image

    Returns:
        array_like: Crops the segmentation mask at the bounding box area
    """
    if yolo:
        x, y, w, h = box[1:-1]
    else:
        x, y, w, h = box[1:]
    x *= img_width
    y *= img_height
    w *= img_width
    h *= img_height

    # Convert to top-left corner coordinates
    x1 = x - w / 2
    y1 = y - h / 2

    if seg_masks is not None:
        mask = seg_masks[:, int(y1):int(y1+h), int(x1):int(x1+w)]
        return mask
    else:       
        return None