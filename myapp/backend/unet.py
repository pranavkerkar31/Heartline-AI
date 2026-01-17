import os # accesing the folder (directories)
import cv2 #image resizing
import torch # engine that runs ai
import numpy as np # handling the data points in the grid
from PIL import Image # handles openieng and saving of the image files
from torchvision import transforms # prepares the images to be udnerstood by the AI

def clean_ecg_pipeline(img_path,model):
    # step 1
    img = cv2.imread(img_path)
     # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    #step 2 brightness augmentation (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) # tilegridsize to make the image 8x8 (64 grids in the ecg paper)
    #clipLimit to limit the brightness of each grid (64 grids)
    enhanced_gray=clahe.apply(gray)

    # step 3 prepare the image for deep learning model
    orig_w,orig_h = enhanced_gray.shape[1],enhanced_gray.shape[0] # stores the original image size which can be retrieved later