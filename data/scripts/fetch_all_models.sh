#!/bin/bash

ls

# Fetch ImageNet model
./data/scripts/fetch_imagenet_model.sh

# Fetch Objectnet Network-Specific model
./data/scripts/fetch_objectnet_network-specific_model.sh

# Fetch Objectnet Network-Specific model
./data/scripts/fetch_pascal3Dplus_network-specific_model.sh
