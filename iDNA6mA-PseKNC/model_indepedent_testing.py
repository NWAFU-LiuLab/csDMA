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
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def exit_with_help(argv):
	print("""\
Usage: python {0} test_positive_dataset test_negative_dataset model_file scale_file
Evaluate the performance of iDNA6mA-PseKNC on the indepedent testing dataset.
The model_file and scale_file generated in the training process must be involved.
""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=5:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    model_file=joblib.load(argv[3])
    scale_file=joblib.load(argv[4])
    return posCV, negCV, model_file, scale_file
        
def main(argv=sys.argv):
    pos_test_file, neg_test_file, model_file, scale_file=process_options(argv)
    feature_matrix=[]
    label_vector=[]
    for line in pos_test_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        feature_vector.extend(PseKNC_code(sequence))
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_test_file.close()
    
    for line in neg_test_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        feature_vector.extend(PseKNC_code(sequence))
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
    print('iDNA6mA-PseKNC Accuracy:%s'%ACC)
    print('iDNA6mA-PseKNC Sensitive:%s'%Sn)
    print('iDNA6mA-PseKNC Specificity:%s'%Sp)
    predictions_test = prob_predict_y_test[:, 1]
    y_validation=np.array(y_test,dtype=int)
    fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
    roc_auc = auc(fpr, tpr)
        
    F1=metrics.f1_score(y_validation, map(int,predict_y_test))
    MCC=metrics.matthews_corrcoef(y_validation,map(int,predict_y_test))
    print('iDNA6mA-PseKNC AUC:%s'%roc_auc)
    print('iDNA6mA-PseKNC F1:%s'%F1)
    print('iDNA6mA-PseKNC MCC:%s'%MCC)
    np.savetxt("iDNA6mA-PseKNC_score.txt",predictions_test,fmt='%s',delimiter='\n')
    np.savetxt("test_label.txt",y_validation,fmt='%s',delimiter='\n')  

if __name__=='__main__':
    main(sys.argv)