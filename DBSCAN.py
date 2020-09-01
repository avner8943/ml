# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:30:56 2020

@author: avner
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy import linalg

def regionQuery(points,p, eps):
    
    dists = linalg.norm(points[:,:4]-p[:4], axis =1)
    
    return points[dists <= eps,:]

def expandCluster(p, sphere_points, cluster, eps, min_points):
    
    for pp in sphere_points:
        
        if pp[4] == 0:
            pp[4] = 1
           
            sphere_points_2 = regionQuery(points, pp, eps)
            
            if (len(sphere_points_2) >= min_points):
                sphere_points = np.append(sphere_points, sphere_points_2, axis = 0)
                t = np.unique(sphere_points, axis = 0)
                
            if np.isnan(pp[5])  or pp[5] == 0:
                pp[5] = cluster
        
    
def DBSCAN(points, eps, min_points):
    cluster = 1
    for p in points:
        
        if p[4] != 0:
            continue
        
        p[4] = 1
        
        sphere_points = regionQuery(points,p,eps)
        
        if (len(sphere_points) < min_points):
            
            p[5] = 0 # mark point as noise
            continue
        
        p[5] = cluster
        
        expandCluster(p, sphere_points, cluster, eps, min_points)
        
        cluster += 1
    
    


if __name__ == "__main__":

    iris_data = load_iris(return_X_y=True)

    points = iris_data[::2]
    
    points = points[0]
    
    points = np.c_[points,np.zeros((150,1)),np.zeros((150,1))]
    
    
    eps = 0.3
    
    min_points = 3
    
    DBSCAN(points, eps,min_points)
