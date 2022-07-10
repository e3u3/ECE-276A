#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:30:32 2022

@author: parth
"""

import cv2
import os
import numpy as np
from gaussian_discriminant_analysis import GDA
from logistic_regression import LR
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label

def classify_test(img):
    # Classify
    labels = model.classify(img.reshape(-1,3))
    #labels = (labels + 1)%2
    mask = labels.reshape(img.shape[0], img.shape[1])

    return mask

# Load Model
model = GDA()#LR() # GDA()
toSave = False
path = '/media/parth/92a679db-c607-4958-bcd2-2db6a6275cf5/home/parth/work/UCSD/Winter 2022/ECE 276/PR1/ECE276A_PR1/bin_detection/results_report'

# Read test image
for idx in range(61,71):
    img_path = 'data/validation/00' + str(idx) + '.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path_ = os.path.join(path, str(idx) + '_1.png')
    
    # Plot image
    plt.imshow(img)
    plt.axis('off')
    if toSave:
        plt.savefig(path_, bbox_inches='tight')
    plt.show()
    
    # Clasiify image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = classify_test(img/255)
    
    # Plot mask
    plt.imshow(mask, cmap = 'gray')
    plt.axis('off')  
    path_ = os.path.join(path, str(idx) + '_2.png')
    if toSave:
        plt.savefig(path_, bbox_inches='tight')
    plt.show() 
    
    # Erosion
    k = int(img.shape[0]/50)
    kernel = np.ones((k,k))
    mask_p = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_ERODE, kernel)
    
    fig, ax = plt.subplots()
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.imshow(mask, cmap = 'gray')
    #plt.imshow(img)
    labels = label(mask_p)
    regions = regionprops(labels)
    bbox = []
    image_area = img.shape[0]*img.shape[1]
    
    for props in regions:   
        minr, minc, maxr, maxc = props.bbox
        w = maxc - minc
        h = maxr - minr
        AR = h/w
        if (AR < 1 or props.area/image_area*100 < 0.1):
            #print(AR)
            continue
        #print(AR, props.area/image_area*100)
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-r', linewidth=2.5)
        bbox.append([minr, minc, maxr, maxc])
    print(bbox)
    plt.axis('off')
    path_ = os.path.join(path, str(idx) + '_3.png')
    if toSave:
        plt.savefig(path_, bbox_inches='tight')
    plt.show()     
    #print(len(bbox))
