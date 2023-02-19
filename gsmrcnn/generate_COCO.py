import sys
sys.path.append("../data")

from OCID_grasp.OCID_class_dict import cnames

import os
import cv2
import numpy as np
from pycocotools import mask
from skimage import measure
import json
from tqdm import tqdm

''' This file is used to create a JSON annotation file (in COCO format) from the OCID grasp dataset. '''


def generate_COCO_from_OCID(dataset_path, split="training", save_path=None):
    """ Produces a COCO formatted JSON file from the OCID grasp dataset and saves it in the dataset's 
    root directory.

    @param dataset_path: (str) path to OCID grasp dataset root directory.
    @param split: (str) whether to generate annotations for the 'training' or 'validation' data splits.
    @param save_path: (str, Optional) path where to save JSON file, if left at None the file will be saved
                      in the 'dataset_path' folder.
    """
    print(f'[INFO] Creating "{split}" annotations for the OCID grasp dataset (dataset_path: "{dataset_path}") ...') 
    annot = {'info':{}, 'licenses':[], 'images':[], 'categories':[], 'annotations':[]} 

    # -----------------  set info parameters -----------------
    annot['info']['decription'] = "Object Clutter Indoor Dataset (OCID) with Grasps"
    annot['info']['url'] = "https://github.com/stefan-ainetter/grasp_det_seg_cnn"
    annot['info']['version'] = "1.0"
    annot['info']['year'] = 2019
    annot['info']['contributor'] = ""
    annot['info']['date_created'] = ""

    # -------------------  set categories ---------------------
    for name, i in cnames.items():
        if i == '0':  # skip 'background' class
            continue
        category = {
            'id': int(i),
            'name': name,
            'supercategory': name
        }
        annot['categories'].append(category)

    # ---------------- set images and annotations ---------------
    # get the file paths for the chosen data split (val or train)
    with open(os.path.join(dataset_path, "data_split", f"{split}_0.txt")) as f:
        file_paths = f.readlines()

    instance_counter = 0  # keeps track of number of object instances in a scene
    grasp_counter = 0  # keeps track of the number of for each object instance
    pbar = tqdm(file_paths, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
    for i, l in enumerate(pbar): # loops through each file path
        path = l.split(',')[0] # the folder path
        name = l.split(',')[1].strip() # the RGB file name
        pbar.set_description(f"[INFO] Processing {name}")

        rgb_path = os.path.join(path, "rgb", name)  # path to RGB image from dataset root directory
        mask_path =  os.path.join(dataset_path, path, "seg_mask_instances_combi", name)  # complete path to instance mask
        class_path = os.path.join(dataset_path, path, "labels.txt") # complete path to image class labels

        img_mask = cv2.imread(os.path.join(mask_path), cv2.IMREAD_GRAYSCALE) # open instance mask
        colours = np.unique(img_mask.flatten())[1:].astype(np.uint8) # get all the unique objects (i.e. colours)

        # open and parse image class labels file into a list
        # note that classes of each object are given in order they are added to scene
        with open(class_path) as f:
            labels = f.readlines()
            labels = labels[0].strip().split(',')

        # for each object instance in the scene
        for pos in range(len(colours)):
            instance_counter += 1
            # get a binary mask of the object
            new_mask = img_mask.copy()
            new_mask[new_mask==colours[pos]] = 255
            new_mask[new_mask!=255] = 0
            new_mask = new_mask.astype(np.uint8)
            # compute the area, bbox and contours from the binary mask
            fortran_gt_binary_mask = np.asfortranarray(new_mask)
            encoded_ground_truth = mask.encode(fortran_gt_binary_mask)
            area = mask.area(encoded_ground_truth)
            bbox = mask.toBbox(encoded_ground_truth).tolist()
            contours = measure.find_contours(new_mask, 0.5)
            category_id = int(labels[colours[pos]-1])
            
            # get grasps
            grasp_path = os.path.join(dataset_path, path, "Annotations_per_class", os.path.splitext(name)[0], str(category_id), os.path.splitext(name)[0] + '.txt')
            grasps = parse_grasps(grasp_path)
            # now filter by instance (i.e check if grasp is this particular object not just class by checking if it is within the bbox)
            x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
            if len(grasps) > 0:
                grasps = [g for g in grasps if (x1 <= g[0] <= x2 and y1 <= g[1] <= y2)]
                grasps_id = [grasp_counter+g+1 for g in range(len(grasps))]
                grasp_counter += len(grasps)
            else:
                grasps_id = []

            annotation = {
                'id': instance_counter,
                'image_id': i + 1,
                'category_id': category_id,
                'bbox': bbox, # [x,y,w,h]
                'area': area,
                'grasps': grasps, # [cx,cy,w,h,t]
                'grasps_id': grasps_id,
                'segmentation': [],
                'iscrowd': 0
            }
            # add contours in Polygon format
            for contour in contours:
                contour = np.round(np.flip(contour, axis=1), 2)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
            annot['annotations'].append(annotation)

        # add image instance
        image = {
            'id': i + 1,
            'file_name': rgb_path,
            'width': 640,
            'height': 480
        }
        annot['images'].append(image)
    
    if save_path is None:
        save_path = dataset_path
    save_path = os.path.join(save_path, f"{split}_annotations.json")
    print(f'[INFO] Saving "{split}" annotation file at "{save_path}".' ) 
    with open(save_path, "w") as outfile:
        json.dump(annot, outfile, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    """ Transforms JSON data from dictionaries into acceptable datatypes if needed (e.g. numpy). """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_5D_pose(bbox):
        """ Given four (x,y) vertices of a grasp rectangle, returns the (x, y, w, h, t) parameters of the grasp pose.
        Note that references include https://www.sciencedirect.com/science/article/pii/S0921889021000427 and
        https://github.com/skumra/robotic-grasping/blob/master/utils/dataset_processing/grasp.py.
        
        @param bbox: (list, tuple) a grasp rectangle as a list of four vertices where each vertex is in (x, y) format.
        @return: (tuple) a tuple (x, y, w, h, t) denoting the 'x, y' centre point, 'w' width, 'h' height and
                 't' rotation of the given grasp rectangle. Note that theta is calculated to be in the range [-pi/2, pi/2].
       """
        x1, x2, x3, x4 = bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]
        y1, y2, y3, y4 = bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]
        cx, cy = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        h = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        w = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
        theta = (np.arctan2((y2 - y1), (x2 - x1))) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
        return round(cx, 3), round(cy, 3), round(w, 3), round(h, 3), round(theta, 5)

def parse_grasps(grasp_path):
    """ Reads a grasp annotation file in Cornell dataset-type format, where each grasp is annoted in four lines representing
        four vertices in 'x, y' format. Each grasp is parsed and converted into a list of grasps of (cx, cy, w, h, t) format. 
    
        @param grasp_path: (str) path to the grasp annotation file to be open and parsed.
        @return: (list, tuple) a list of grasps in (cx, cy, w, h, t) format, where theta is in the range [-pi/2, pi/2].
    """
    grasps = []
    with open(grasp_path) as f:
        grasp_annots = f.readlines()
        grasp_rect = []  # to store the vertices of a single grasp rectangle
        for i, l in enumerate(grasp_annots):
            # parse the (x,y) co-ordinates of each grasp box vertice
            xy = l.strip().split()
            grasp_rect.append((float(xy[0]), float(xy[1])))
            if (i + 1) % 4 == 0:
                if not np.isnan(grasp_rect).any():
                    cx, cy, w, h, theta = convert_to_5D_pose(grasp_rect)
                    grasps.append((cx, cy, w, h, theta))
                    grasp_rect = []  # reset current grasp rectangle after 4 vertices have been read
    return grasps

if __name__ == '__main__':
    dataset_path = "../data/OCID_grasp"
    save_path = "../data"
    generate_COCO_from_OCID(dataset_path, "validation", save_path=save_path)  # "training" or "validation"

    
    