
from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import random 
import pandas as pd



def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 


def get_anchors(centroids, width_in_cfg_file, height_in_cfg_file): 
    anchors = centroids.copy()
    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file
        anchors[i][1] *= height_in_cfg_file
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)
    return anchors, sorted_indices, anchors[sorted_indices]


def avg_IOU(X, centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file, width_in_cfg_file,
                          height_in_cfg_file):
    
    anchors , sorted_indices, anchors_sort = get_anchors(centroids,
                                                         width_in_cfg_file,
                                                         height_in_cfg_file)
    print('Anchors = ', anchors_sort)
        
#    for i in sorted_indices[:-1]:
#        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))
#
#    #there should not be comma after last anchor, that's why
#    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
#    
#    f.write('%f\n'%(avg_IOU(X,centroids)))
    return anchors_sort


def kmeans(X,centroids,eps,anchor_file, width_in_cfg_file, height_in_cfg_file):
    
    N = X.shape[0]
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            anchors = write_anchors_to_file(centroids,X,anchor_file,
                                            width_in_cfg_file,
                                            height_in_cfg_file)
            return anchors

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     


def generate_anchor(label_csv, width_in_cfg_file, height_in_cfg_file, 
                    num_clusters=3 ):


    f = pd.read_csv(label_csv)

    annotation_dims = np.array([tuple(x) for x in f.iloc[:,4:].values])
  
    eps = 0.005
    
    if num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join('anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            anchors = kmeans(annotation_dims,centroids,eps,anchor_file,
                             width_in_cfg_file, height_in_cfg_file)

    else:
        anchor_file = join('anchors%d.txt'%(num_clusters))
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        anchors = kmeans(annotation_dims,centroids,eps,anchor_file,
                         width_in_cfg_file, height_in_cfg_file)
    return anchors


def set_anchors_to_model(model, num_anchors, label_csv_mame, input_w, input_h):
    anchors = generate_anchor(label_csv_mame, input_w, input_h,
                              num_clusters=num_anchors*3)
    # loop through yolo layer from the back and ad anchors box number to
    # each yolo layer. larger anchors are in the front layers
    mask = []
    start = 0
    num_anchor_per_layer = num_anchors
    for i in range(num_anchors*3+1):
        if i and (i % num_anchor_per_layer) == 0:
            mask.append(np.arange(start, i))
            start = i  
    for index, layer in enumerate(model.layer_type_dic['yolo'][::-1]):
        model.module_list[layer][0].anchors = anchors[mask[index]]
        
if __name__=="__main__":
    label_csv= '../color_balls/label.csv'
    width_in_cfg_file = 512.
    height_in_cfg_file = 512.
    num_clusters = 3
    anchors = generate_anchor(label_csv, width_in_cfg_file, height_in_cfg_file,
                              num_clusters=3)
