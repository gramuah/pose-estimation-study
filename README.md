# The challenge of simultaneous object detection and pose estimation: a comparative study

By Daniel Oñoro-Rubio, Roberto J. López-Sastre, Redondo-Cabrera and Pedro Gil-Jiménez

This is the original implmenentatio of the paper calld *The challenge of simultaneous object detection and pose estimation: a comparative study*.

**Note:** Some of the scripts that populate the models will be included soon.

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Pretrained models](#download-pre-trained-models)
5. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](https://dl.dropboxusercontent.com/s/6joa55k64xo2h68/Makefile.config?dl=0) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`


### Requirements: hardware

1. For training the models is required a GPU with 6GB of memory and CUDA support.


### Installation

1. Clone the 'pose-estimation-study' repository
  ```Shell
  git clone https://github.com/gramuah/pose-estimation-study.git
  ```

2. We'll call the directory that you cloned `PROJECT_ROOT`

 
3. Build the Cython modules
    ```Shell
    cd $PROJECT_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $PROJECT_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

### Download pre-trained models

Pre-trained model can be fetch by running the script:

```Shell
cd $PROJECT_ROOT
./data/scripts/fetch_models.sh
```

### Usage

To train and test a detector use the corresponding script: 

```Shell
cd $PROJECT_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Output is written underneath `$PROJECT_ROOT/output`.


("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.

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
