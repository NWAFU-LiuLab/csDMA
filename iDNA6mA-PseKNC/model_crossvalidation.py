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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from numpy import  *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def exit_with_help(argv):
	print("""\
Usage: python {0} positive_dataset negative_dataset
This script trains the iDNA6mA-PseKNC model.
5-fold cross-validation was used to evaluate the performance of the classifier.""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=3:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    return posCV, negCV
        
def main(argv=sys.argv):
    pos_train_file, neg_train_file=process_options(argv)
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
    
######GridSearchCV
    
#    C_range = np.logspace(-1, 1, 100)
#    gamma_range = np.logspace(-2, 1, 100)
#    param_grid = dict(gamma=gamma_range, C=C_range)
#    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
#    grid.fit(X, y)
#    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
    
    fold_path='../cross_data/fold_five.txt'
    fold_auc=[]
    fold=[]
    fold_file=open(fold_path,'r')
    for line in fold_file:
        fold_temp=line.split()
        fold.append(fold_temp[0])
    fold_file.close()
    num_folds=5  
    SVC_ACC=[]
    SVC_AUC=[]
    SVC_Sn=[]
    SVC_Sp=[]
    SVC_MCC=[]
    SVC_F1=[]        
    for i in range(1,num_folds+1):
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        combined_prob=[]
        for j in range(0,len(fold)):
            if int(fold[j])==i:
                X_test.append(X[j])
                y_test.append(y[j])
            else:
                X_train.append(X[j])
                y_train.append(y[j])
######SVMClassifier
        clf = SVC(C=0.336,gamma=0.02,probability=True)
        clf.fit(X_train,y_train)
        r = clf.score(X_test,y_test)
        #print('SVMClassifier Mean Accuracy:%s'%r)
        SVC_ACC.append(r)
        predict_y_test = clf.predict(X_test)
        
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
        SVC_Sn.append(Sn)
        SVC_Sp.append(Sp)        
        
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]
        y_validation=np.array(y_test,dtype=int)
        fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        roc_auc = auc(fpr, tpr)
        #print('SVMClassifier AUC:%s'%roc_auc)
        SVC_AUC.append(roc_auc)
        
        F1=metrics.f1_score(y_validation, map(int,predict_y_test))
        MCC=metrics.matthews_corrcoef(y_validation,map(int,predict_y_test))
        SVC_F1.append(F1)
        SVC_MCC.append(MCC)
          
    print('SVMClassifier Mean Accuracy:%s'%mean(SVC_ACC))
    print('SVMClassifier AUC:%s'%mean(SVC_AUC))
    print('SVMClassifier Mean Sensitive:%s'%mean(SVC_Sn))
    print('SVMClassifier Mean Specificity:%s'%mean(SVC_Sp))
    print('SVMClassifier Mean F1:%s'%mean(SVC_F1))
    print('SVMClassifier Mean MCC:%s'%mean(SVC_MCC))
      
if __name__=='__main__':
    main(sys.argv)
