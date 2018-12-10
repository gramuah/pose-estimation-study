#!/bin/bash

ls

# Fetch ImageNet model
./data/scripts/fetch_imagenet_model.sh

mkdir data/imagenet_models/
mv data/VGG16.v2.caffemodel data/imagenet_models/VGG16.v2.caffemodel

# Fetch Objectnet Network-Specific model
./data/scripts/fetch_objectnet_network-specific_model.sh

mkdir data/demo_models/
mv data/SPECIFIC-NETWORK_OBJECTNET.caffemodel data/demo/SPECIFIC-NETWORK_OBJECTNET.caffemodel


# Fetch Objectnet Network-Specific model
./data/scripts/fetch_pascal3Dplus_network-specific_model.sh

mv data/SPECIFIC-NETWORK_3DPLUS.caffemodel data/demo_models/SPECIFIC-NETWORK_3DPLUS.caffemodel
