# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps

class PoseLabelGenerator(caffe.Layer):
    """
    Generate the correct labels for each pose classifier. It will return the pose
    labels for the current class, and -1 (ignored label) for the other objects.
    """

    def setup(self, bottom, top):
        # Pose labels CLS = 1
        top[0].reshape(1, 1)
        # Pose labels CLS = 2
        top[1].reshape(1, 1)
        # Pose labels CLS = 3
        top[2].reshape(1, 1)
        # Pose labels CLS = 4
        top[3].reshape(1, 1)
        # Pose labels CLS = 5
        top[4].reshape(1, 1)
        # Pose labels CLS = 6
        top[5].reshape(1, 1)
        # Pose labels CLS = 7
        top[6].reshape(1, 1)
        # Pose labels CLS = 8
        top[7].reshape(1, 1)
        # Pose labels CLS = 9
        top[8].reshape(1, 1)
        # Pose labels CLS = 10
        top[9].reshape(1, 1)
        # Pose labels CLS = 11
        top[10].reshape(1, 1)
        # Pose labels CLS = 12
        top[11].reshape(1, 1)
        
    def forward(self, bottom, top):

        # Get foreground labels
        fg_labels = bottom[0].data
        # Get azimuths gt
        fg_azimuths = bottom[1].data

        for ix in range(1,13):
            # Get labels match
            label_match = fg_labels == ix
            pose_labels = np.ones(fg_labels.shape, np.float32)*-1
            
            pose_labels[label_match] = fg_azimuths[label_match, ix]

            top[ix-1].reshape(*pose_labels.shape)
            top[ix-1].data[...] = pose_labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass