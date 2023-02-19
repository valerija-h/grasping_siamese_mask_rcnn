import sys
sys.path.append('../libraries/') # add path to Siamese Mask R-CNN library
sys.path.append('../libraries/Mask_RCNN') # add path to Mask R-CNN library

from Siamese_Mask_RCNN.lib import config as smrcnn_config

# GSM-RCNN file for training on the OCID grasp dataset
class OCIDConfig(smrcnn_config.Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 1 + 1 # background (not same as target) + object (same as target)
    NAME = 'object_split' # model name
    EXPERIMENT = '1' # experiment no.
    CHECKPOINT_DIR = 'models/'
    IMAGE_MIN_DIM = 480 # image height
    IMAGE_MAX_DIM = 640 # image width
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0,
                    'mrcnn_grasp_loss': 0.8}
    NUM_TARGETS = 1 # number of references/target images to use 
    MAX_GT_GRASPS = 10 # maximum number of GT grasps to sample for each object instance
    MAX_GT_INSTANCES = 5 # the number of ground-truth object instances per image
    STEPS_PER_EPOCH = 150
    VALIDATION_STEPS = 10

# GSM-RCNN config file for evaluating the OCID grasp dataset
class OCIDConfigEvaluation(OCIDConfig):
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9 # confidence threshold value - lower values may be better for newer objects

if __name__ == "__main__":
    config = OCIDConfig()
    config.display()
