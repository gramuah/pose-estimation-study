# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred_3Dplus'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred_3Dplus'][0].data.copy()
            orig_1 = net.params['bbox_pred_3Dplus'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred_3Dplus'][0].data[...] = \
                    (net.params['bbox_pred_3Dplus'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred_3Dplus'][1].data[...] = \
                    (net.params['bbox_pred_3Dplus'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred_3Dplus'][0].data[...] = orig_0
            net.params['bbox_pred_3Dplus'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        
#         iter_v = []
#         cls_grad_v = []
#         bbx_grad_v = []
#         pose_grad_v = []
#         
#         cls_w_v = []
#         bbx_w_v = []
#         pose_w_v = []
        
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

#             # Capture gradients
#             if (self.solver.iter % 20) == 0:
#                 net = self.solver.net
#                  
#                 cls_grad = np.mean(net.blobs['cls_score_3Dplus'].diff, axis = 0)
#                 cls_grad = np.linalg.norm(cls_grad)
#                 print "Cls grad:", cls_grad
#                  
#                 bbx_grad = np.mean(net.blobs['bbox_pred_3Dplus'].diff, axis = 0)
#                 bbx_grad = np.linalg.norm(bbx_grad)
#                 print "BBX grad:", bbx_grad
#                  
#                 pose_grad = np.mean(net.blobs['pose_pred_3Dplus'].diff, axis = 0)
#                 pose_grad = np.linalg.norm(pose_grad)
#                 print "pose grad:", pose_grad
#  
#                 iter_v.append(self.solver.iter)
#                 cls_grad_v.append(cls_grad)
#                 bbx_grad_v.append(bbx_grad)
#                 pose_grad_v.append(pose_grad)
#  
#                 cls_w_v.append( net.params['cls_score_3Dplus'][0].data[-1,-1] )
#                 bbx_w_v.append( net.params['bbox_pred_3Dplus'][0].data[-1,-1] )
#                 pose_w_v.append( net.params['pose_pred_3Dplus'][0].data[-1,-1] )

            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
