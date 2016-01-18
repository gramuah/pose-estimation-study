'''
Created on Jan 5, 2016

@author: Daniel Onoro Rubio
'''


import datasets
import datasets.pascal_voc
import os
import glob
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from statvfs import F_NAMEMAX

class pascal_3Dplus(datasets.imdb):
    def __init__(self, pascal3Dplus_path=None, pascal_path=None):
        datasets.imdb.__init__(self, '3Dplus')
        self._year = 2013
        self._image_set = '3Dplus' 
        self._pascal3Dplus_path = self._get_default_path() if pascal3Dplus_path is None \
                            else pascal3Dplus_path
        self._pascal_path = self._get_pascal_default_path() if pascal_path is None \
                            else pascal_path
        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'boat',
                         'bottle', 'bus', 'car', 'chair',
                         'diningtable', 'motorbike', 'sofa',
                         'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._pascal3Dplus_path), \
                'Pascal path does not exist: {}'.format(self._pascal3Dplus_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        
        #is Imagenet?
        if 'imagenet' in index:
            im_ext = ".JPEG" 
        else:
            im_ext = ".jpg"
        
        image_path = os.path.join(self._pascal3Dplus_path, 'Images',
                                  index + im_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the training and validation image names.
        """
        # Example path to image set file:
        # self._pascal3Dplus_path + /PACAL3D+/Image_sets/aeroplane_imagenet_train.txt
        imagenet_set_folder = os.path.join(self._pascal3Dplus_path, 'Image_sets')
        imagenet_set_train_files = glob.glob(os.path.join(imagenet_set_folder, '*_train.txt'))
        imagenet_set_val_files = glob.glob(os.path.join(imagenet_set_folder, '*_val.txt'))
        
        # self._pascal3Dplus_path + /PACAL3D+/Image_sets/aeroplane_imagenet_train.txt
        pacal_set_folder = os.path.join(self._pascal_path, 'ImageSets', 'Main')

        # self._pascal3Dplus_path + PASCAL/VOCdevkit/VOC2012/ImageSets/Main/bicycle_trainval.txt'
        pascal_set_trainval_files = []
        for i in range(1, len( self._classes )):
            im_set = os.path.join(pacal_set_folder, '{}_trainval.txt'.format(self._classes[i])) 
            pascal_set_trainval_files.append(im_set) 
        
        # Concatenate train and val
        image_set_train_val_files = imagenet_set_train_files + \
                imagenet_set_val_files + pascal_set_trainval_files 
        
        image_names = []
        for f_name in image_set_train_val_files:
            assert os.path.exists(f_name), \
                    'Path does not exist: {}'.format(f_name)

            # Get folder name            
            last_slash = f_name.rfind(os.path.sep) + 1
            last_ = f_name.rfind('_')
            folder_name = f_name[last_slash:last_] if "imagenet" in f_name else f_name[last_slash:last_] + "_pascal"  
            
            with open(f_name) as f:
                # Read for imagenet notation
                if "imagenet" in f_name:
                    image_index = [os.path.join( folder_name, x.strip() ) for x in f.readlines() ]
                else:
                # Read for pascal notation
                    image_index = []
                    for x in f.readlines():
                        aux = x.strip() # remove end of line
                        aux = aux.split(' ', 2)
                        # Check class flag
                        if aux[-1] == '1':
                            image_index.append( os.path.join( folder_name, aux[0] ) )
            
            image_names += image_index
            
        return image_names

    def _get_default_path(self):
        """
        Return the default path where PASCAL 3D PLUS is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'PASCAL3D+')

    def _get_pascal_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(self._pascal3Dplus_path, 'PASCAL', 'VOCdevkit', 'VOC2012')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from .mat file in the PASCAL3D
        format.
        """
        filename = os.path.join(self._pascal3Dplus_path, 'Annotations', index + '.mat')
        raw_data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
        objs = raw_data['record'].objects if hasattr(raw_data['record'].objects, '__iter__') else [raw_data['record'].objects] 
        im_size = raw_data['record'].size
        
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs
                             if int( obj.difficult ) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects' \
                    .format(len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        mask = np.zeros((num_objs), dtype=np.bool )

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            aux_cls = str( getattr(obj, 'class') )
            # Check whether the object belong to our collection. If not, ignore
            mask[ix] = self._class_to_ind.has_key(aux_cls)
            if not mask[ix]:
                continue
            cls = self._class_to_ind[ aux_cls ]
            
            bbox = obj.bbox
            # Make pixel indexes 0-based
            x1 = float( bbox[0] ) - 1
            y1 = float( bbox[1] ) - 1
            x2 = float( bbox[2] ) - 1
            y2 = float( bbox[3] ) - 1

            # Rectify boxes to fit withing the image. PASCAL3D+ contains bbox out of the image range!!
            aux_box = np.zeros_like(bbox)
            aux_box[0] = max(x1, 0)
            aux_box[1] = max(y1, 0)
            aux_box[2] = min(x2, im_size.width - 1)
            aux_box[3] = min(y2, im_size.height - 1)
            boxes[ix, :] = np.asarray( aux_box, np.uint16 )
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        # Filter out classes
        overlaps = overlaps[mask]
        boxes = boxes[mask]
        gt_classes = gt_classes[mask]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # Create results folder if not exist
        results_path = os.path.join(self._get_default_path(), 'results')
        if not os.path.exists( results_path ):
            os.makedirs( results_path )

        # PASCAL3D+/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(results_path, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._pascal3Dplus_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_3Dplus()
    res = d.roidb
    from IPython import embed; embed()
