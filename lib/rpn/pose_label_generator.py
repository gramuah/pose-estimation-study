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
import datasets.pose_utils as utls

class PoseLabelGenerator(caffe.Layer):
    """
    Generate the correct labels for each pose classifier. It will return the pose
    labels for the current class, and -1 (ignored label) for the other objects.
    """

    def setup(self, bottom, top):
        """Setup the PoseLabelGenerator."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._num_bins = layer_params['num_bins']
        self._pose_interval = utls.generate_interval(self._num_bins)

        # Pose labels for Azimuth
        top[0].reshape(1, 1)      
        # Pose labels for Elevation
        top[1].reshape(1, 1)
        # Pose labels for Theta
        top[2].reshape(1, 1)
        # Weights for pose
        top[3].reshape(1, 1)
         
    def forward(self, bottom, top):
        # Get foreground labels
        class_labels = bottom[0].data
        # Get azimuths gt
        fg_azimuths = bottom[1].data
        fg_elevations = bottom[2].data
        fg_thetas = bottom[3].data

        # Get labels
        azimuth_labels = np.zeros( (fg_azimuths.shape[0], 1), dtype=np.float32 )
        elevation_labels = np.zeros( (fg_elevations.shape[0], 1), dtype=np.float32 )
        theta_labels = np.zeros( (fg_thetas.shape[0], 1), dtype=np.float32 )
        pose_weights = np.zeros( (fg_azimuths.shape[0], self._num_classes * self._num_bins), dtype=np.float32 ) 
        # Split fg from bg        
        bg_ix = np.where( class_labels == 0)[0]
        fg_ix = np.where( class_labels > 0)[0]
        # Prepare data
        azimuth_labels[bg_ix] = -1.0
        elevation_labels[bg_ix] = -1.0
        theta_labels[bg_ix] = -1.0
        for ix in fg_ix:
            class_offset = int(self._num_bins*class_labels[ix])
            azimuth_labels[ix] = float(utls.find_interval(fg_azimuths[ix], self._pose_interval) + class_offset)
            pose_weights[ix, class_offset:class_offset + self._num_bins] = 1.0
            elevation_labels[ix] = utls.find_interval(fg_elevations[ix], self._pose_interval) + class_offset
            theta_labels[ix] = utls.find_interval(fg_thetas[ix], self._pose_interval) + class_offset
        
        # Forward labels
        top[0].reshape(*azimuth_labels.shape)
        top[0].data[...] = azimuth_labels
        top[1].reshape(*elevation_labels.shape)
        top[1].data[...] = elevation_labels
        top[2].reshape(*theta_labels.shape)
        top[2].data[...] = theta_labels
        top[3].reshape(*pose_weights.shape)
        top[3].data[...] = pose_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class ContinuousPoseLabelGenerator(caffe.Layer):
    """
    Generate the TODO...
    """
    def setup(self, bottom, top):
        """Setup the PoseLabelGenerator."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        # Pose azimuth labels
        top[0].reshape(1, self._num_classes*2)
        # Pose elevation labels
        top[1].reshape(1, self._num_classes*2)
        # Pose theta labels
        top[2].reshape(1, self._num_classes*2)
        # Pose weights labels
        top[3].reshape(1, self._num_classes*2)
        
    def forward(self, bottom, top):
        # Get foreground labels
        class_labels = bottom[0].data
        # Get azimuths gt
        fg_azimuths = bottom[1].data
        fg_elevations = bottom[2].data
        fg_thetas = bottom[3].data
        # Get pose and weights
        azimuth_labels, _ = self._get_pose_regression_labels(fg_azimuths, class_labels, self._num_classes)
        elevation_labels, _ = self._get_pose_regression_labels(fg_elevations, class_labels, self._num_classes)
        theta_labels, weights_vector = self._get_pose_regression_labels(fg_thetas, class_labels, self._num_classes)
        # Forward data
        top[0].reshape(*azimuth_labels.shape)
        top[0].data[...] = azimuth_labels
        top[1].reshape(*elevation_labels.shape)
        top[1].data[...] = elevation_labels
        top[2].reshape(*theta_labels.shape)
        top[2].data[...] = theta_labels
        top[3].reshape(*weights_vector.shape)
        top[3].data[...] = weights_vector

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
    
    def _get_pose_regression_labels(self, azimuth, labels, num_classes):
        """
        This function expands the azimuth targets into the N*2*K representation used
        by the network (i.e. only one class has non-zero targets).
        
        The angles are decomposed into cos and sin.
    
        Returns:
            pose_target (ndarray): N x 2K blob of regression targets
            pose_inside_weights (ndarray): N x 2K blob of loss weights
        """
    
        pose_targets = np.zeros((azimuth.shape[0], 2 * num_classes), dtype=np.float32)
        pose_inside_weights = np.zeros(pose_targets.shape, dtype=np.float32)
        inds = np.where(labels > 0)[0]
        for ind in inds:
            cls = labels[ind].astype(np.int32)
            start = 2 * cls
            end = start + 2
            # Cast pose into radians
            r_pose = azimuth[ind] * np.pi / 180.0
            pose_targets[ind, start:end] = [np.cos(r_pose), np.sin(r_pose)] 
            pose_inside_weights[ind, start:end] = [1.0, 1.0]
        return pose_targets, pose_inside_weights
