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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from numpy import  *

def exit_with_help(argv):
	print("""\
Usage: python {0} training_positive_dataset training_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file
The csDMA trained with the ExtraTrees classifier.
Outputs:
     1--a model file, csDMA.pkl, which can be directly used for prediction.
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.
""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=7:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    matrix_motif1_file=open(argv[3],'r')
    matrix_motif2_file=open(argv[4],'r')
    model=argv[5]
    scale=argv[6]
    return posCV, negCV, matrix_motif1_file, matrix_motif2_file, model, scale
        
def main(argv=sys.argv):
    pos_train_file, neg_train_file, matrix_motif1_file, matrix_motif2_file, model_file, scale_file=process_options(argv)

######import motif1 matrix    
    matrix_motif1=[]
    for line in matrix_motif1_file:
        matrix_motif1_vector=line.split()
        matrix_motif1.append(matrix_motif1_vector)
    matrix_motif1_file.close()
    matrix1_array_T=map(list,zip(*matrix_motif1))
    matrix1_array=np.array(matrix1_array_T,dtype=np.float32) 

######import motif2 matrix
    matrix_motif2=[]
    for line in matrix_motif2_file:
        matrix_motif2_vector=line.split()
        matrix_motif2.append(matrix_motif2_vector)
    matrix_motif2_file.close()
    matrix2_array_T=map(list,zip(*matrix_motif2))
    matrix2_array=np.array(matrix2_array_T,dtype=np.float32)     
######train feature extraction
    feature_matrix=[]
    label_vector=[]
    for line in pos_train_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        #feature_vector.extend(ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence))
        matrix1_score=matrix_motif1_func(sequence,matrix1_array)
        feature_vector.append(matrix1_score)
        matrix2_score=matrix_motif2_func(sequence,matrix2_array)
        feature_vector.extend(matrix2_score)
        feature_vector.extend(kmer(sequence)+binary_code(sequence))
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_train_file.close()
    
    for line in neg_train_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        #feature_vector.extend(ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence))
        matrix1_score=matrix_motif1_func(sequence,matrix1_array)
        feature_vector.append(matrix1_score)
        matrix2_score=matrix_motif2_func(sequence,matrix2_array)
        feature_vector.extend(matrix2_score)
        feature_vector.extend(kmer(sequence)+binary_code(sequence))
        label_vector.append('-1')
        feature_matrix.append(feature_vector)
    feature_array = np.array(feature_matrix,dtype=np.float32)
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= min_max_scaler.fit_transform(feature_array)
    neg_train_file.close()

    X=feature_scaled
    y=label_vector
######extraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
    #clf = RandomForestClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
    #clf = GradientBoostingClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
    #clf = AdaBoostClassifier(n_estimators=1000)
    #clf = SVC(C=0.98,gamma=0.01,probability=True)
    clf.fit(X,y)
    joblib.dump(clf,model_file)
    joblib.dump(min_max_scaler,scale_file)

if __name__=='__main__':
    main(sys.argv)