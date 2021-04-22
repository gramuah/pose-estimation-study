# The challenge of simultaneous object detection and pose estimation: a comparative study

By Daniel Oñoro-Rubio, Roberto J. López-Sastre, Carolina Redondo-Cabrera and Pedro Gil-Jiménez.


This is a repository with the original implementation of the object detection and pose estimation solutions described in our [IMAVIS  journal paper](https://doi.org/10.1016/j.imavis.2018.09.013). 


### License

This repository is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you make use of this data and software, please cite the following reference in any publications:

    @Article{Onoro-Rubio2018,
    author     = {O\~noro-Rubio, D. and L\'opez-Sastre, R.~J. and Redondo-Cabrera, C. and Gil-Jim\'enez, P.},
    title   = {The challenge of simultaneous object detection and pose estimation: a comparative study},
    journal = {IMAVIS},
    year    = {2018},
    volume = {79},
    pages = {109-122},
    issn = {0262-8856},
    doi = {https://doi.org/10.1016/j.imavis.2018.09.013},
    }


## Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Pre-trained models](#download-pre-trained-models)
5. [Usage](#usage)
6. [Demo: test](#demo-test)

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

1. For training, the models are required a GPU with at least 6 GB of memory and CUDA support.


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

Step 5: Downloading datasets

In order to test or to train any of the models included in this repository, it is needed one of the following datasets:

* PASCAL3D+: [http://cvgl.stanford.edu/projects/pascal3d.html](http://cvgl.stanford.edu/projects/pascal3d.html)
* ObjectNet3D: [http://cvgl.stanford.edu/projects/objectnet3d](http://cvgl.stanford.edu/projects/objectnet3d)

The datasets must be manually downloaded from the author's platform and placed in the `data` directory.

## Download pre-trained models

Direct links to download the pre-trained models can be accessed in the file [data/scripts/README.md](data/scripts/README.md).

## Usage

This repository includes the code needed to perform a complete training and testing of any of the proposed models (i.e.: `Single-path`, `Specific-path`, and `Specific-network`) by using the PASCAL3D+ or the ObjectNet3D. All the running scripts are located in: `experiments/scripts/<experiment>.sh`.

As an example, to train and test our `Single-path` on the ObjectNet3D, just execute: 

    ```Shell
    cd $PROJECT_ROOT
    ./experiments/scripts/objectnet_single-path.sh [GPU_ID]
    # GPU_ID is the GPU you want to train on
    ```

The output is written underneath `$PROJECT_ROOT/output`.


Trained models are saved under:

    ```
    output/<experiment directory>/<dataset name>/
    ```

Test outputs are saved under:

    ```
    output/<experiment directory>/<dataset name>/<network snapshot name>/
    ```

## Demo: test

We provide the pre-trained models of our SPECIFIC-NETWORK model, hence it is possible to run the test.

To run the test on the Pascal3DPlus just execute: 

    ```Shell
    cd $PROJECT_ROOT
    ./experiments/scripts/demo_pascal_3D_network-specific [GPU_ID]
    # GPU_ID is the GPU you want to train on
    ```

Or to run it for the ObjectNet3D, execute: 

    ```Shell
    cd $PROJECT_ROOT
    ./experiments/scripts/demo_objectnet_network-specific.sh [GPU_ID]
    # GPU_ID is the GPU you want to train on
    ```
