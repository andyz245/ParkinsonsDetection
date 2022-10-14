# Specific Installtion Requirements

Instructions can be found in INSTALL.md

## MMpose

MMpose is necessary to preprocess the videos.

conda create --name mmpose --clone opence-v1.5.1

conda activate mmpose

pip cache purge

pip install --upgrade pip

pip install --upgrade pip setuptools

git clone --recursive https://github.com/opencv/opencv-python.git

cd opencv-python

pip wheel . --verbose

cd ~

git clone https://github.com/open-mmlab/mmcv.git

cd mmcv

pip install -r requirements/optional.txt

MMCV_WITH_OPS=1 pip install -e .

cd ~

git clone https://github.com/open-mmlab/mmpose.git

cd mmpose

#COMMENT OUT "onnxruntime" and "albumentations>=0.3.2 --no-binary qudida,albumentations" in requirements/optional.txt

pip install -r requirements.txt

pip install -v -e .

pip install mmdet

pip install geos

cd ~/.conda/envs/mmpose/lib/

ln -s /usr/lib64/libgeos_c.so.1.11.2 libgeos_c.so

cd ~/mmpose

## Data
Videos can be downloaded with the script in the data folder, based on the labels in the spreadsheet. 

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).


## Contributors
Andy Zhou, Sam Li, Pranav Sriram, Yuanyi Zhong, Xiang Li, Yuxiong Wang
