# Preparing the Data
Arcface trains on variety of face data in the paper. For my implementation, I have decided to train the network with 
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and validate the data with [CFP-FP] dataset. 
I chose these two datasets since they had the least amount of data out of the datasets in [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
which the original authors of Arcface recommended on their [Github page](https://github.com/deepinsight/insightface). 
Therefore, the data preprocessing code is compatible with CelebA data, hence the name "crop_celeb_a.py".

##Requirements
[MTCNN](https://pypi.org/project/mtcnn/) and [OpenCV](https://pypi.org/project/opencv-python/) was used to perform face detection and image cropping respectively.
Note that MTCNN requires Tensorflow run, therefore install Tensorflow if you already have not. 


## Training Data
### 1. Downloading the Dataset
Download the *img_align_celeba_png.7z* from [Multimedia Lab, The Chinese University of Hong Kong's Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

### 2. Run code
```bash
python crop_celeb_a.py 
```
There are arguments you can add.
```bash
--main_dir "<main_directory>" 
--src_dir "<directory_of_source_images>" 
--tar_dir "<directory of where target images will go>" 
--txtfile "identity_CelebA.txt"
--continue "<image number to contine from>"
--img_size "<size of output image>"
```
There are default parameters implemented based on the celebA dataset.

After the code is ran, the directory tree should look something like this
```buildoutcfg
arcface_sample
└── train
    ├── 1
    │   ├── 000023.jpg
    │   ├── 004506.jpg
    │   ├── 006439.jpg
    │   ├── ...
    ├── 2
    │   ├── 016188.jpg
    │   ├── 051523.jpg
    │   ├── 068490.jpg
    │   ├── ...
    ├── 3
    │   ├── 003029.jpg
    │   ├── 003206.jpg
    │   ├── 008838.jpg
    │   ├── ...
    ├── 4
    │   ├── 008268.jpg
    │   ├── 022543.jpg
    │   ├── 022852.jpg
    │   ├── ...
    ├── 5
    │   ├── 000456.jpg
    │   ├── 018771.jpg
    │   ├── 038782.jpg
    │   ├── ...
    ├── 6
    │   ├── 002617.jpg
    │   ├── 003289.jpg
    │   ├── 004317.jpg
    │   ├── ...
    └── ...
```
where each directory under train are folders with each identity (1-10177 for CelebA dataset), and files in each folders are pictures of each identity.
Put the absolute path of the top directory into config.yaml file under the field *train_faces_path*.

## Validation Data
### 1. Downloading the Dataset
Download the *CFPW dataset* from [CFPW dataset page](http://www.cfpw.io/)

### 2. Extract the file
Extract the file on the same directory as the training data is residing.

### 3. Done
The reading and processing of the data is done on [dataloader/dataloader.py](https://github.com/chaddy1004/arcface/blob/master/dataloader/dataloader.py) 


