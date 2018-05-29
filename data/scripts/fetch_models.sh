#!/bin/bash

ls

# Fetch ImageNet model
./data/scripts/fetch_imagenet_model.sh

# Fetch Objectnet Network-Specific_(trainval-test) model
./data/scripts/fetch_network-specific_model.sh