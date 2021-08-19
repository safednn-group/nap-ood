#!/bin/bash

DATABASE='data'

if [ ! -d "$DATABASE" ]; then

mkdir $DATABASE
pushd $DATABASE
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
unzip GTSRB-Training_fixed.zip -d  GTSRB-Training_fixed
rm GTSRB-Training_fixed.zip


wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Online-Test-Images-Sorted.zip
unzip GTSRB_Online-Test-Images-Sorted.zip -d GTSRB_Online-Test-Images-Sorted
rm GTSRB_Online-Test-Images-Sorted.zip 

popd

fi
# install dd package from tar.gz
#python -m pip install dd-0.5.4.tar.gz 

