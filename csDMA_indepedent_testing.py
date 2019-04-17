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
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def exit_with_help(argv):
	print("""\
Usage: python {0} test_positive_dataset test_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file
Evaluate the performance of csDMA on the indepedent testing dataset.
The model_file and scale_file generated in the training process must be involved.
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
    model_file=joblib.load(argv[5])
    scale_file=joblib.load(argv[6])
    return posCV, negCV, matrix_motif1_file, matrix_motif2_file, model_file, scale_file
        
def main(argv=sys.argv):
    pos_test_file, neg_test_file, matrix_motif1_file, matrix_motif2_file, model_file, scale_file=process_options(argv)

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
    for line in pos_test_file:
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
    pos_test_file.close()
    
    for line in neg_test_file:
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
    #min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= scale_file.transform(feature_array)
    neg_test_file.close()

    X_test=feature_scaled
    y_test=label_vector
######extraTreesClassifier
#    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
#    clf.fit(X,y)
#    joblib.dump(clf,model_file)
#    joblib.dump(min_max_scaler,scale_file)
    predict_y_test = model_file.predict(X_test)
    prob_predict_y_test = model_file.predict_proba(X_test)
    TP=0
    TN=0
    FP=0
    FN=0 
    for i in range(0,len(y_test)):
        if int(y_test[i])==1 and int(predict_y_test[i])==1:
            TP=TP+1
        elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
            FN=FN+1
        elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
            TN=TN+1
        elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
            FP=FP+1
    Sn=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP)
    ACC=float((TP+TN))/(TP+TN+FP+FN)
    print('ExtraTrees Accuracy:%s'%ACC)
    print('ExtraTrees Sensitive:%s'%Sn)
    print('ExtraTrees Specificity:%s'%Sp)
    predictions_test = prob_predict_y_test[:, 1]
    y_validation=np.array(y_test,dtype=int)
    fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
    roc_auc = auc(fpr, tpr)
        
    F1=metrics.f1_score(y_validation, map(int,predict_y_test))
    MCC=metrics.matthews_corrcoef(y_validation,map(int,predict_y_test))
    print('ExtraTrees AUC:%s'%roc_auc)
    print('ExtraTrees F1:%s'%F1)
    print('ExtraTrees MCC:%s'%MCC)
    np.savetxt("csDMA_score.txt",predictions_test,fmt='%s',delimiter='\n')
    np.savetxt("test_label.txt",y_validation,fmt='%s',delimiter='\n')  

if __name__=='__main__':
    main(sys.argv)