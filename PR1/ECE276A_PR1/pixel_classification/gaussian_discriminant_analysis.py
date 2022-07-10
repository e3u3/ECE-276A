#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:45:39 2022

@author: parth
"""
import numpy as np
import os
import pickle
from generate_rgb_data import read_pixels

class GDA:
    def __init__(self):
        '''
        Constructor for class GDA.
        Initializes model from file. If not present, performs 
        training to get model
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        GDA_Model_path = os.path.join(dir_path,'GDA_Model.pickle')
        
        if os.path.exists(GDA_Model_path):
            print('Loading Model from file')
            with open('GDA_Model.pickle', 'rb') as handle:
                GDA_Model = pickle.load(handle)
        else:
            GDA_Model = self.train()
            with open(r"GDA_Model.pickle", "wb") as handle:
                pickle.dump(GDA_Model, handle)
            
        self.Model = GDA_Model

        
    def train(self):
        '''
        Function fits a Gaussian model on each 
        class.
        Output:
            Dict with c entries each having the
            model parameter (list)
        '''
        # Get training data
        print('Training')
        DATA = self.read_training_data()        
        # Dict to save model for each class
        model_params = {}
        total = 0
        for color in DATA:
            data = DATA[color]
            # Calculate model parameters
            mean = np.mean(data, axis = 0).reshape(1,-1)
            cov  = np.matmul((data-mean).T, (data-mean))/data.shape[0]
            prior = data.shape[0]
            total = total + data.shape[0]
            model_params[color] = [prior, mean, cov]
        
        # Normalize class prioirs to probability values
        for color in DATA:
            model_params[color][0] = model_params[color][0]/total 

        return model_params
    
    def classify(self, X):
        '''
        Given a nx3 test data X, function returns a 
        nx1 label vector
        {1,2,3} - > {R,G,B}
        '''
        scores = []
        for c in self.Model:
            prior = self.Model[c][0] 
            mean = self.Model[c][1]
            cov = self.Model[c][2]
            
            # Gaussian Calculation
            X_ = X-mean
            ML = np.multiply(X_ @ np.linalg.inv(cov), X_)/2
            ML = np.sum(ML,1)
            score = np.exp(-ML)
            d = cov.shape[0]
            denom = np.sqrt(np.power(2*np.pi,d)*np.linalg.det(cov))
            score = score/denom
            
            score = score * prior
            scores.append(score)
        scores = np.array(scores)
        y = np.argmax(scores, 0)+1
        
        return y
    
    def read_training_data(self):
        '''
        Returns a dict with key corresponding to color and 
        value numpy array as training data
        Output:
            Dictionary with key corresponding to each color.
        '''
        path = 'data/training'
        DATA = {}
        for c in os.listdir(path):
            path_ = os.path.join(path,c)
            data_ = read_pixels(path_, False)
            DATA[c] = data_
        return DATA