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

class LR:
    def __init__(self):
        '''
        Constructor for class GDA.
        Initializes model from file. If not present, performs 
        training to get model
        '''
        
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
        '''
        LD_Model = self.train()
        self.Model = LD_Model


    def softmax(self, X):
        X_ = np.exp(X)
        return X_/np.sum(X_,1).reshape(-1,1)
        
    
    def computeCost(self, X, y, W):
        '''
        Compute cost
        '''
        scores = -np.log(self.softmax(X @ W.T))
        cost = np.multiply(scores, y)
        cost = np.sum(cost)/X.shape[0]
        return cost
        
    
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
        X,y = self.read_training_data()        
        
        # Initialize weights (kxd)
        W = np.random.rand(3,3)
        
        # max iter
        max_iters = 100
        
        # learning rate
        alpha = 0.1
        
        for i in range(max_iters):
            # Not sure abput this vectorization
            gradient = (y - self.softmax(X @ W.T)).T @ X
            W = W + alpha*gradient        
            cost = self.computeCost(X,y,W)
            print(cost)
            if cost < 0.001:
                break
        
        return W
    
    def classify(self, X):
        '''
        Given a nx3 test data X, function returns a 
        nx1 label vector
        {1,2,3} - > {R,G,B}
        '''
        W = self.Model
        scores = self.softmax(X@W.T)
        y = np.argmax(scores,1)+1
        return y
    
    def read_training_data(self):
        '''
        Returns a dict with key corresponding to color and 
        value numpy array as training data
        Output:
            Dictionary with key corresponding to each color.
        '''
        path = 'data/training'
        X = []
        y = []
        for key,c in enumerate(os.listdir(path)):
            path_ = os.path.join(path,c)
            data_ = read_pixels(path_, False)
            X.append(data_)
            y_ = np.zeros((data_.shape[0],3))
            y_[:,key] = 1
            y.append(y_)
        X = np.concatenate(X,0)
        y = np.concatenate(y,0)
        return X,y
