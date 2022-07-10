'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import os
import pickle

from gaussian_discriminant_analysis import GDA
from LogisticRegression import LR

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''        
    self.Model = GDA()
    #self.Model = LR()
    pass
      
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    
    y = self.Model.classify(X)
      
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

