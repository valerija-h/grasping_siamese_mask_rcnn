import sys
import os
import numpy as np
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import model as modellib
from shapely.geometry import Polygon as Shapely_Polygon
from skimage.measure import find_contours
import datetime

from pycocotools import mask as maskUtils

sys.path.append('../libraries/')
from Siamese_Mask_RCNN.lib import utils as smrcnn_utils
from Mask_RCNN.mrcnn import utils as mrcnn_utils
from Mask_RCNN.mrcnn import visualize as mrcnn_visualize

sys.path.append('../libraries/Mask_RCNN')
from samples.coco import coco


''' This file contains a custom data class and functions to evaluate the GSM-RCNN model. '''

############################################################
#  Dataset Object
############################################################

class OCIDDataset(smrcnn_utils.IndexedCocoDataset):
    """ The same class as the COCO dataset but I inputted the .json and img_dir manually. """
    def load_coco(self, annotation_json, image_dir, class_ids=None, return_coco=False):
        """ Load a subset of the OCID dataset in COCO format. This function was changed
        from the original to just load the JSON file directly.

        annotation_json: Name of the annotation file to load.
        image_dir: Path to root of OCID image directory.
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        # CHANGE: load annotation file
        coco = COCO(annotation_json)

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco
    
    # CHANGE: Created a function to load grasps for a given image
    def load_grasps(self, image_id, MAX_GT_GRASPS=10):
        """ Load instance grasps for the given image. Note many can be 0 - these will be ignored.
            
        image_id: An int ID of the image to load.
        MAX_GT_GRASPS: The maximum number of grasps to sample for each object.
        
        Returns:
        grasps: Array of shape [num_instances, MAX_GT_GRASPS, (x, y, w, h, t)].
        """
        annotations = self.image_info[image_id]["annotations"]
        n = len(annotations)

        grasps = np.zeros((n, MAX_GT_GRASPS, 5), dtype=np.float32)
        for i in range(n):
            annotation = annotations[i]
            grasp_list = np.asarray(annotation['grasps'])
            if grasp_list.shape[0] > 0:
                # if the grasp list is too large, sample MAX_GT_GRASPS
                if grasp_list.shape[0] > MAX_GT_GRASPS:
                    idxs = np.random.choice(np.arange(grasp_list.shape[0]), MAX_GT_GRASPS, replace=False)
                    grasp_list = grasp_list[idxs]
                grasps[i, :grasp_list.shape[0]] = grasp_list
        return grasps
    
    def build_indices(self):
        """ Creates dictionaries sorting image ids to category ids.
        
        image_category_index: A list of shape [NUM_IMGS, X], where X is the number of classes present in an image.
        category_image_index: A list of shape [NUM_CLASSES, X], where X is the number of images that contain a class.
        """
        self.image_category_index = smrcnn_utils.IndexedCocoDataset._build_image_category_index(self)
        self.category_image_index = OCIDDataset._build_category_image_index(self.image_category_index)

    def _build_category_image_index(image_category_index):
        category_image_index = []
        # CHANGE: Fix
        for category in range(max(map(max, image_category_index))+1):
            # Find all images corresponding to the selected class/category 
            images_per_category = np.where(\
                [any(image_category_index[i][j] == category\
                 for j in range(len(image_category_index[i])))\
                 for i in range(len(image_category_index))])[0]
            # Put list together
            category_image_index.append(images_per_category)

        return category_image_index


def get_one_target(category, dataset, config, augmentation=None, target_size_limit=0, max_attempts=10, return_all=False,
                   return_original_size=False, image_id=None, apply_mask=False, custom_augmentation=False):
    """ Retrieves a reference/target image for a specified category. """
    n_attempts = 0
    while True:
        # Get index with corresponding images for each category
        category_image_index = dataset.category_image_index

        # Draw a random image
        if image_id is None:
            random_image_id = np.random.choice(category_image_index[category])
        else:
            random_image_id = image_id

        # Load image
        target_image, target_image_meta, target_class_ids, target_boxes, target_masks, _ = \
            modellib.load_image_gt(dataset, config, random_image_id, augmentation=augmentation)

        if not np.any(target_class_ids == category):
            continue

        box_ind = np.random.choice(np.where(target_class_ids == category)[0])
        tb = target_boxes[box_ind, :]

        if apply_mask:
            mask = np.repeat(target_masks[:,: ,box_ind][:, :, np.newaxis], 3, axis=2) * 1
            target_image = target_image * mask

        target = target_image[tb[0]:tb[2], tb[1]:tb[3], :]
        original_size = target.shape

        target, window, scale, padding, crop = mrcnn_utils.resize_image(
            target,
            min_dim=config.TARGET_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,  # Same scaling as the image
            max_dim=config.TARGET_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)  # Same output format as the image

        if custom_augmentation:
            # 50% chance of flip image
            if random.random() < 0.5:
                target = np.fliplr(target)
            # 50% chance of rotating image
            if random.random() < 0.5:
                if random.random() < 0.5:
                    target = np.rot90(target, 1)
                else:
                    target = np.rot90(target, -1)


        n_attempts = n_attempts + 1
        if (min(original_size[:2]) >= target_size_limit) or (n_attempts >= max_attempts):
            break

    if return_all:
        return target, window, scale, padding, crop
    elif return_original_size:
        return target, original_size
    else:
        return target
    
############################################################
#  COCO Evaluation Scripts
############################################################

# CHANGE: Add grasps to results object
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks, grasps):
    """ Arrange results to match COCO specs in http://cocodataset.org/#format. The function
     takes in predictions from the GSM-RCNN model. """
    
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]
            grasp = grasps[i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask)),
                "grasp": grasp
            }
            results.append(result)
    return results


def evaluate_dataset(model, dataset, dataset_object, eval_type="bbox", dataset_type='coco',
                     limit=0, image_ids=None, class_index=None, verbose=1, random_detections=False,
                     return_results=False):
    """ Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    assert dataset_type in ['coco']
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    dataset_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        if i % 10 == 0 and verbose > 1:
            print("Processing image {}/{} ...".format(i, len(image_ids)))

        # Load GT data
        # CHANGE: Added grasps
        _, _, gt_class_ids, _, _, grasps = modellib.load_image_gt(dataset, model.config,
                                                          image_id, augmentation=False,
                                                          use_mini_mask=model.config.USE_MINI_MASK)
        # BOILERPLATE: Code duplicated in siamese_data_loader

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            continue

        # Use only positive class_ids
        categories = np.unique(gt_class_ids)
        _idx = categories > 0
        categories = categories[_idx]
        # Use only active classes
        active_categories = []
        for c in categories:
            if any(c == dataset.ACTIVE_CLASSES):
                active_categories.append(c)

        # Skiop image if it contains no instance of any active class
        if not np.any(np.array(active_categories) > 0):
            continue

        # END BOILERPLATE

        # Evaluate for every category individually
        for category in active_categories:

            # CHANGE: Skip if it doesn't have grasps
            idx = gt_class_ids == category
            c_grasps = grasps[idx] # Get grasps for current category
            missing_grasps = False
            for g in c_grasps:
                if not np.any(g):
                    missing_grasps = True  # True if one of the target instances is missing grasps...
            if missing_grasps:
                continue

            # Load image
            image = dataset.load_image(image_id)

            # Draw random target
            target = []
            for k in range(model.config.NUM_TARGETS):
                try:
                    target.append(get_one_target(category, dataset, model.config, image_id=image_id, apply_mask=True))
                except:
                    print('error fetching target of category', category)
                    continue
            target = np.stack(target, axis=0)
            # Run detection
            t = time.time()
            try:
                r = model.detect([target], [image], verbose=0, random_detections=random_detections)[0]
            except:
                print('error running detection for category', category)
                continue
            t_prediction += (time.time() - t)

            # Format detections
            r["class_ids"] = np.array([category for i in range(r["class_ids"].shape[0])])

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            # CHANGE: Added grasps
            if dataset_type == 'coco':
                image_results = build_coco_results(dataset, dataset_image_ids[i:i + 1],
                                                        r["rois"], r["class_ids"],
                                                        r["scores"],
                                                        r["masks"].astype(np.uint8), r["grasps"])
            results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    dataset_results = dataset_object.loadRes(results)

    # allow evaluating bbox & segm:
    if not isinstance(eval_type, (list,)):
        eval_type = [eval_type]

    seg_results, bbox_results = None, None
    for current_eval_type in eval_type:
        # Evaluate
        cocoEval = customCOCOeval(dataset_object, dataset_results, current_eval_type)
        cocoEval.params.imgIds = dataset_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize(class_index=class_index, verbose=verbose)
        if verbose > 0:
            print("Prediction time: {}. Average {}/image".format(
                t_prediction, t_prediction / len(image_ids)))
            print("Total time: ", time.time() - t_start)

        if current_eval_type == 'bbox':
            bbox_results = cocoEval
        if current_eval_type == 'segm':
            seg_results = cocoEval

    if return_results:
        return [bbox_results, seg_results]


# Change: Need to overwrite COCO evaluator to put grasp scores as well.
class customCOCOeval(COCOeval):

    def summarize(self, class_index=None, verbose=1):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f} | grasp={:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                if not class_index is None:
                    s = s[:, :, class_index, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:, class_index, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            # CHANGE: Evaluate the grasps
            g = self.eval['grasp_scores']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                g = g[t]
            if not class_index is None:
                g = g[:, class_index, aind, mind]
            else:
                g = g[:, :, aind, mind]
            if len(g[g > -1]) == 0:
                mean_g = -1
            else:
                mean_g = np.mean(g[g > -1])

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            # CHANGE: Added grasps
            if verbose > 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s, mean_g))
            return [mean_s, mean_g]

        def _summarizeDets():
            stats = np.zeros((12,2))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self, class_index=None):
        self.summarize(class_index)


    def evaluateImg(self, imgId, catId, aRng, maxDet):
        ''' Perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range (aRng) to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        # note - for a single image we are collecting information for each unique category instance

        # CHANGE:
        # get first IOU threshold to match each TP grasp to a GT grasp
        # TODO - find a more optimised way of doing this for each threshold without recomputing grasps...
        # right now we use the gt_grasp from the first threshold is consistent with the rest
        t0 = dtm[0]  # first IOU threshold
        found_grasps = []
        for i, gt_id in enumerate(t0):  # for each detected grasp
            f_grasp = 0
            if gt_id != 0:  # if detected object was matched to a GT object
                det_grasp = dt[i]['grasp']  # current detection grasp
                gt_grasps = [g['grasps'] for g in gt if g['id'] == gt_id][0]  # gt grasps of matched gt object
                # determine whether any of the grasp are accurate
                f_grasp = check_valid_grasp(det_grasp, gt_grasps)
            found_grasps.append(f_grasp)

        final_found_grasps = np.asarray([[found_grasps[i] if gt_id != 0 else 0 for i, gt_id in enumerate(t)] for t in dtm])

        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'gtGrasps':     [g['grasps'] for g in gt], # CHANGE: added GT grasps and pred grasp for each instance in image (GT/DT is a list of instances in an image)
                'dtGrasp':      [d['grasp'] for d in dt],
                'dtGraspMatches': final_found_grasps # same shape as dtm signifying if grasp was success or not
            }

    # follow explanation here https://github.com/cocodataset/cocoapi/blob/master/MatlabAPI/CocoEval.m
    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs) # IOU threshold
        R           = len(p.recThrs) # recall threshold
        K           = len(p.catIds) if p.useCats else 1 # no. of categories
        A           = len(p.areaRng) # area scope
        M           = len(p.maxDets) # maximum number of detections
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        grasp_scores= -np.ones((T,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK] # categories
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM] # max detections
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA] # areas
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI] #img list
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list): # == 4 [0, 1, 2, 3] referring to [[[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
                Na = a0*I0
                for m, maxDet in enumerate(m_list): # == 3 [1, 10, 100]
                    # E == the evaluate results of Image in current k0 (cat) and current a0 (area range)
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    # dtMatches - [T x D] whether there is a matching GT id at each IOU threshold T = 10
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    gtGrasps = np.concatenate([e['dtGraspMatches'][:,0:maxDet]  for e in E], axis=1)[:,inds]

                    npig = np.count_nonzero(gtIg==0)
                    if npig == 0:
                        continue

                    # tps are ones that have a match and are not ignored (i.e. not part of this area)
                    # fps are ones that have no match and are not ignored
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    gps = np.logical_and(               gtGrasps,  np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    grasp_sum = np.cumsum(gps, axis=1).astype(dtype=np.float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig  # no. of TP computed for the given IoU threshold / no. of GT boxes
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        # CHANGE: added grasp score
                        grasp_score = grasp_sum[t]/tp
                        if nd and tp[-1] != 0:  # only calculate grasp scores on true positives
                            grasp_scores[t,k,a,m] = grasp_score[-1]
                        else:
                            grasp_scores[t,k,a,m] = -1  # ignore if there are no true positives...

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'grasp_scores': grasp_scores
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))


def get_points(bbox, t):
    """ Get all four vertices (x,y) of a rotated rectangle, when given a bbox with rotation t. """
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    x, y = xmax - (w / 2), ymax - (h / 2)

    w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
            h / 2) * np.cos(t)
    bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
    br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
    return (tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y)

def check_valid_grasp(pred_grasp, gt_grasps):
    """ Checks whether a predicted grasp is valid. """
    theta_pred = pred_grasp[4]
    bbox_pred = pred_grasp[:4]
    for gt in gt_grasps:
        gt_theta = gt[4]
        gt_bbox = gt[:4]
        # check if theta is within 30 degrees
        if np.abs(gt_theta - theta_pred) < 0.523599 or (np.abs(np.abs(gt_theta - theta_pred) - np.pi)) < 0.523599:
            # now check if IOU > 0.25
            gt_grasp = Shapely_Polygon(get_points(gt_bbox, gt_theta))
            pred_grasp = Shapely_Polygon(get_points(bbox_pred, theta_pred))
            intersection = gt_grasp.intersection(pred_grasp).area / gt_grasp.union(pred_grasp).area
            if intersection > 0.25:
                return 1
    return 0


############################################################
#  Visualization
############################################################

def display_results(target, image, boxes, masks, class_ids, grasps=None,
                    scores=None, title="",
                    figsize=(16, 16), ax=None,
                    show_mask=True, show_bbox=True,
                    colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        from matplotlib.gridspec import GridSpec
        # Use GridSpec to show target smaller than image
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3)
        ax = plt.subplot(gs[:, 1:])
        target_ax = plt.subplot(gs[1, 0])
        auto_show = True

    # Generate random colors
    colors = colors or mrcnn_visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    target_height, target_width = target.shape[:2]
    target_ax.set_ylim(target_height + 10, -10)
    target_ax.set_xlim(-10, target_width + 10)
    target_ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = mrcnn_visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                            alpha=0.7, linestyle="dashed",
                                            edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # CHANGE: Grasp
        if not np.any(grasps[i]):
            # Skip this instance. Has no bbox. Likely lost in cropping.
            continue

        x, y, w, h, t = grasps[i]
        w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
                h / 2) * np.cos(t)
        bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
        br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
        ax.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
        ax.plot([br_x, tr_x], [br_y, tr_y], c='black')
        rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor=color, facecolor='none',
                                transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
        ax.add_patch(rect)


        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{:.3f}".format(score) if score else 'no score'
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = mrcnn_visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = mrcnn_visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = mrcnn_visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    target_ax.imshow(target.astype(np.uint8))
    if auto_show:
        plt.show()

def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None, grasps=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = mrcnn_visualize.random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Grasps
        if grasps is not None:
            if not np.any(grasps[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            for g in grasps[i]:
                if not np.any(g):
                    # Skip this instance. Has no bbox. Likely lost in cropping.
                    continue
                x, y, w, h, t = g
                w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
                        h / 2) * np.cos(t)
                bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
                br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
                plt.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
                plt.plot([br_x, tr_x], [br_y, tr_y], c='black')
                rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor=color, facecolor='none',
                                        transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
                ax.add_patch(rect)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = mrcnn_visualize.apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
