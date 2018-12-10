# The challenge of simultaneous object detection and pose estimation: a comparative study

By Daniel Oñoro-Rubio, Roberto J. López-Sastre, Carolina Redondo-Cabrera and Pedro Gil-Jiménez.


This is a repository with the original implementation of the object detection and pose estimation solutions described in our [IMAVIS  journal paper](https://arxiv.org/abs/1801.08110). 


### License

This repository is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you make use of this data and software, please cite the following reference in any publications:

	@Article{Onoro-Rubio2018,
	author 	= {O\~noro-Rubio, D. and L\'opez-Sastre, R.~J. and Redondo-Cabrera, C. and Gil-Jim\'enez, P.},
	title   = {The challenge of simultaneous object detection and pose estimation: a comparative study},
	journal = {IMAVIS},
	year    = {2018},
	}


## Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Pre-trained models](#download-pre-trained-models)
5. [Usage](#usage)

## Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # It's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download our [Makefile.config](https://dl.dropboxusercontent.com/s/6joa55k64xo2h68/Makefile.config?dl=1) for reference.

2. Python packages you also need: `cython`, `python-opencv`, `easydict`


## Requirements: hardware

1. For training the models is required a GPU with at least 6 GB of memory and CUDA support.


## Installation

Step 1:  Clone this repository

  ```Shell
  git clone https://github.com/gramuah/pose-estimation-study.git
  ```

Step 2: We'll call the directory that you cloned `PROJECT_ROOT`

 
Step 3: Build the Cython modules

    ```Shell
    cd $PROJECT_ROOT/lib
    make
    ```

Step 4: Build Caffe and pycaffe


    ```Shell
    cd $PROJECT_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

## Download pre-trained models

Pre-trained models can be obtained by running the following script:

```Shell
cd $PROJECT_ROOT
./data/scripts/fetch_models.sh
```

## Usage

To train and test our simultaneous object detection and pose estimation models use the corresponding scripts: 

```Shell
cd $PROJECT_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [--set ...]
# GPU_ID is the GPU you want to train on
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Output is written underneath `$PROJECT_ROOT/output`.


```Shell
cd $PROJECT_ROOT
./experiments/scripts/objectnet_single-path.sh [GPU_ID]
```

Output is written underneath `$PROJECT_ROOT/output`.


Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
