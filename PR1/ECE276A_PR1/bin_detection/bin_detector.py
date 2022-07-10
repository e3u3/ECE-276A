'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import regionprops,label
from gaussian_discriminant_analysis import GDA
#from logistic_regression import LR

class BinDetector():
    def __init__(self):
        '''
        Initilize your bin detector with the attributes you need,
        e.g., parameters of your classifier
        '''
        self.model = GDA() # LR()

    def segment_image(self, img):
        '''
        Obtain a segmented image using a color classifier,
        e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
        call other functions in this class if needed

        Inputs:
                img - original image
        Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Pre-process
        print('Segmenting image ...')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)/255
        labels = self.model.classify(img.reshape(-1, 3))
        mask_img = labels.reshape(img.shape[0], img.shape[1])
    
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return mask_img

    def get_bounding_boxes(self, img):
        '''
        Find the bounding boxes of the recycling bins
        call other functions in this class if needed

        Inputs:
                img - original image
        Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        
        print('Detect bounding boxes ...')
        k = 11 #int(img.shape[0]/50)
        image_area = img.shape[0]*img.shape[1]
        kernel = np.ones((k, k))
        mask_p = cv2.morphologyEx(img.astype('uint8'), cv2.MORPH_ERODE, kernel)
    
        labels = label(mask_p)
        regions = regionprops(labels)
    
        boxes = []
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            w = maxc - minc
            h = maxr - minr
            AR = h/w
            if (AR < 1 or props.area/image_area*100 < 0.1):
                continue
            boxes.append([minc, minr, maxc, maxr])
        
        print('Detected bounding boxes' + str(len(boxes)))
        # YOUR CODE BEFORE THIS LINE
        ################################################################

        return boxes
