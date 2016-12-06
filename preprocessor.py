# -*- coding: utf-8 -*-

import cv2
import numpy as np

def to_grayscale(X):
  w = np.array([[0.299], [0.587], [0.114]])
  return X.dot(w)

  
#augment the training data using translation and rotation
def augment_data(X, y):
  num_X = X.shape[0]
  delta_std=3
  X_trans=np.empty_like(X)
  y_trans=np.empty_like(y)
  #translation
  for i in range(num_X):
    delta_x, delta_y = np.random.randint(-delta_std,delta_std+1, 2)
    M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    X_trans[i] = cv2.warpAffine(X[i],M,(32,32))[:,:, np.newaxis]
    y_trans[i] = y[i]
  
  theta_std=3
  X_rot=np.empty_like(X)
  y_rot=np.empty_like(y)
  #translation
  for i in range(num_X):
    theta = np.random.randn()*theta_std
    M = cv2.getRotationMatrix2D((16,16),theta,1)
    X_rot[i] = cv2.warpAffine(X[i],M,(32,32))[:,:, np.newaxis]
    y_rot[i] = y[i]
  
  X = np.concatenate((X, X_trans, X_rot), axis=0)
  y = np.concatenate((y, y_trans, y_rot), axis=0)
  return (X,y)

##normalization
def normalize(X):
  return (X-128)/128

##one-hot encoding
def one_hot_encoding(y,n_classes):
  num_y = y.shape[0]
  _,y_grid=np.mgrid[0:num_y,0:n_classes]
  y_tile = np.tile(y.reshape(num_y,1), (1,n_classes))
  y_one_hot= np.zeros_like(y_grid)
  y_one_hot[y_grid==y_tile]=1
  return y_one_hot
