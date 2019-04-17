#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:11:20 2018

@author: liuze
"""
import sys
from sklearn.externals import joblib
from feature_selection import *
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from numpy import  *

def exit_with_help(argv):
	print("""\
Usage: python {0} training_positive_dataset training_negative_dataset model_file scale_file
This script was used to implement the iDNA6mA-PseKNC tool.
Outputs:
     1--a model file, iDNA6mA-PseKNC.pkl, which can be directly used for prediction.
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.
""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=5:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    model=argv[3]
    scale=argv[4]
    return posCV, negCV, model, scale
        
def main(argv=sys.argv):
    pos_train_file, neg_train_file, model_file, scale_file=process_options(argv)
    feature_matrix=[]
    label_vector=[]
    for line in pos_train_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        feature_vector.extend(PseKNC_code(sequence))
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_train_file.close()
    
    for line in neg_train_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        feature_vector.extend(PseKNC_code(sequence))
        label_vector.append('-1')
        feature_matrix.append(feature_vector)
    feature_array = np.array(feature_matrix,dtype=np.float32)
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= min_max_scaler.fit_transform(feature_array)
    neg_train_file.close()

    X=feature_scaled
    y=label_vector
######SVMClassifier
    clf = SVC(C=0.336,gamma=0.02,probability=True)
    clf.fit(X,y)
    joblib.dump(clf,model_file)
    joblib.dump(min_max_scaler,scale_file)

if __name__=='__main__':
    main(sys.argv)