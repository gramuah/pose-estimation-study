'''
Created on Jan 8, 2016

@author: Daniel Onoro Rubio
'''

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class PoseLossLayer(caffe.Layer):
    """ 
        Pose loss layer that computes the biternion loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")
        
        self.DEG2RAD_CONST = np.pi/180.0 
        
        # Pose weigths cls*n_bins
        self.pose_weigths = [0.92, 1.06, 0.98, 1.04, 0.97, 1.07, 1.02, 1.06, \
                             0.89, 0.99, 0.99, 1.02, 0.74, 1.10, 1.10, 1.10, \
                             1.00, 1.08, 1.09, 1.01, 0.58, 0.90, 0.93, 0.87, \
                             0.90, 1.07, 1.04, 1.02, 0.81, 1.10, 1.02, 1.03, \
                             0.98, 1.07, 1.03, 1.06, 0.91, 1.10, 1.08, 1.03, \
                             0.95, 1.09, 1.09, 0.99, 0.91, 1.10, 1.09, 1.03]

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != (bottom[1].count*2):
            raise Exception("Pose prediction does not match with pose labels dimensions.")
        
        # To save inner products between pred and GT
        self.inner_prod = np.zeros( bottom[0].data.shape[0] )
        # Hold predicted modules
        self.pred_mod = np.zeros( bottom[0].data.shape[0] )
        # Hold polar labels
        self.pol_labels = np.zeros( (bottom[0].data.shape[0], 2) )
        
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        '''
        Forward pass: 
            bottom[0]: predicted tuple (unnormalized)
            bottom[1]: pose angle labels (degrees)
            bottom[2]: class labels 
        '''
        cls_labels = bottom[2].data.astype(np.int32)    # Cast them to integer
#         done= False
        total_loss = 0
        inds = np.where(cls_labels > 0)[0]
        for ix in inds:
            cls = cls_labels[ix]
            # Cast labels into polar cordinates (cos x sin x)
            rad_labels = bottom[1].data[ix,cls]*self.DEG2RAD_CONST
            polar_labels = np.hstack( (np.cos(rad_labels), np.sin(rad_labels) ) ).reshape((1,2))
            polar_pred = bottom[0].data[ix, cls*2:cls*2+2].reshape((2,1))
        
            self.pol_labels[ix] = polar_labels
            self.inner_prod[ix] = np.dot(polar_labels,polar_pred)
            self.pred_mod[ix] = np.linalg.norm(polar_pred)
        
            loss = 1 - self.inner_prod[ix]/self.pred_mod[ix]
            
            total_loss += loss
            
#             if not done:
#                 done = True
#                 print "GT: ", polar_labels
#                 print "Pred: ", polar_pred.ravel()/self.pred_mod[ix]
                
        
        top[0].data[...] = total_loss/len(inds)


    def backward(self, top, propagate_down, bottom):
        # Reset gradients
        bottom[0].diff[...] = np.zeros_like(bottom[0].diff)
        
        # Get class labels
        cls_labels = bottom[2].data.astype(np.int32)    # Cast them to integer
#         pose_labels = bottom[1].data    # Poses
#         
#         pose_step = 360.0/4
#         last_bin_angle = 360 - pose_step/2.0
        
        inds = np.where(cls_labels > 0)[0]
        for ix in inds:
            cls = cls_labels[ix]
            # First parameter
            bottom[0].diff[ix, cls*2] += self.inner_prod[ix]*bottom[0].data[ix, cls*2] / (self.pred_mod[ix]**3) \
                -self.pol_labels[ix, 0] / self.pred_mod[ix]
            
            # Second parameter
            bottom[0].diff[ix, cls*2 + 1] += self.inner_prod[ix]*bottom[0].data[ix, cls*2+1] / (self.pred_mod[ix]**3) \
                -self.pol_labels[ix, 1] / self.pred_mod[ix]

            # Weight loss           
#             wcls_ix = (cls-1)*4
#             if pose_labels[ix, cls] > last_bin_angle:
#                 pose_ix = 0
#             else:
#                 pose_ix = int( abs(pose_labels[ix, cls] - pose_step/2)/pose_step ) # Compute class + pose idx
#                 
#             w = self.pose_weigths[wcls_ix + pose_ix]                
#             bottom[0].diff[ix, cls*2:cls*2 + 2] *= w  
    
        
        # Normalize output    
        bottom[0].diff[...] = bottom[0].diff[...] #/ bottom[0].num
        
class PoseEuclideanLossLayer(caffe.Layer):
    """ 
        Pose loss layer that computes the biternion loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")
        
        self.DEG2RAD_CONST = np.pi/180.0 
        
        # Pose weigths cls*n_bins
        self.pose_weigths = [0.92, 1.06, 0.98, 1.04, 0.97, 1.07, 1.02, 1.06, \
                             0.89, 0.99, 0.99, 1.02, 0.74, 1.10, 1.10, 1.10, \
                             1.00, 1.08, 1.09, 1.01, 0.58, 0.90, 0.93, 0.87, \
                             0.90, 1.07, 1.04, 1.02, 0.81, 1.10, 1.02, 1.03, \
                             0.98, 1.07, 1.03, 1.06, 0.91, 1.10, 1.08, 1.03, \
                             0.95, 1.09, 1.09, 0.99, 0.91, 1.10, 1.09, 1.03]

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != (bottom[1].count*2):
            raise Exception("Pose prediction does not match with pose labels dimensions.")
        
        # Hold polar labels
        self.pol_labels = np.zeros( (bottom[0].data.shape[0], 2) )
        
        # Init diff
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        
        # Count of fg objets
        self.count = 1
        
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        '''
        Forward pass: 
            bottom[0]: predicted tuple (unnormalized)
            bottom[1]: pose angle labels (degrees)
            bottom[2]: class labels 
        '''
        # Init diff
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        
        cls_labels = bottom[2].data.astype(np.int32)    # Cast them to integer
        done= False
        inds = np.where(cls_labels > 0)[0]
        self.count = 0
#         for ix in range( len(cls_labels) ):
        for ix in inds:
            cls = cls_labels[ix]
            # Cast labels into polar cordinates (cos x sin x)
            rad_labels = bottom[1].data[ix,cls]*self.DEG2RAD_CONST
            polar_labels = np.hstack( (np.cos(rad_labels), np.sin(rad_labels) ) ).reshape((1,2))
            polar_pred = bottom[0].data[ix, cls*2:cls*2+2].reshape((1,2))
            
#             if not done:
#                 done = True
#                 print "GT: ", polar_labels
#                 print "Pred: ", polar_pred
            
            self.count += 1
            self.diff[ix, cls:cls+2] = polar_pred - polar_labels
        
        self.count = max(self.count,1)
        top[0].data[...] = np.sum(self.diff**2) / self.count / 2.


    def backward(self, top, propagate_down, bottom):
        # Normalize output    
        bottom[0].diff[...] = self.diff[...] / self.count