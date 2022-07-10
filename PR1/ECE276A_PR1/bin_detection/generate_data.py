#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:21:11 2022

@author: parth
"""
import os
import cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt


folder = 'data/training'
img_list = sorted(os.listdir(folder))

MASK_P = []
MASK_NP = []

def save_results(MASK_P, MASK_NP):
    '''
    Saves the annotated data
    '''
    output_folder = 'data_bin'
    print('Saving Data....')
    output_file_name = "Positive.npy"
    MASK_P = np.concatenate(MASK_P,0) 
    print(MASK_P.shape)
    path = os.path.join(output_folder, output_file_name)
    np.save(path, MASK_P)

    output_file_name = "Negative.npy"
    MASK_NP = np.concatenate(MASK_NP,0) 
    print(MASK_NP.shape)
    path = os.path.join(output_folder, output_file_name)
    np.save(path, MASK_NP)
    
    return

def get_average_color(img, mask):
    '''
    Read pixels marked and extract average color value 
    of the pixels
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color = img[mask,:]
    #avg_color = np.mean(color,0)
    #avg_color = avg_color.reshape(1,-1)
    return color#avg_color
    

print('Annotating Data....')
for img_fileName in img_list:
    flag = True
    print(img_fileName) 
    while flag:
        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        img = cv2.imread(os.path.join(folder,img_fileName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        
        ax.imshow(img)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')
        
        # get the image mask
        mask = my_roi.get_mask(img)
        
        # Get label class
        key = input("Enter class key...")
        
        color = get_average_color(img.copy(), mask)
        if key == 'p':
            MASK_P.append(color)
        elif key == 'n':
            MASK_NP.append(color)
        
        # Label another region
        key = input("Mark another region...")
        
        if key =='y':
            flag = True
        else:
            flag = False
    # Save intermediate results
    save_results(MASK_P, MASK_NP)

save_results(MASK_P, MASK_NP)
    