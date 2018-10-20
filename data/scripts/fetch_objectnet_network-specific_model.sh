#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE="SPECIFIC-NETWORK_OBJECTNET.caffemodel"
URL="http://agamenon.tsc.uah.es/Personales/rlopez/data/pose-estimation-study/SPECIFIC-NETWORK_OBJECTNET.caffemodel"
CHECKSUM="c1f516865b8bfbf741ffcf36eb2a5ef2"

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading pretrained SPECIFIC-NETWORK_OBJECTNET.caffemodel model (1.1G)..."
wget $URL -O $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."