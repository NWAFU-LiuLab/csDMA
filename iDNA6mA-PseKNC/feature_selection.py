# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:35:41 2018

@author: Administrator
liuze
"""
from __future__ import division
import sys
from functools import reduce
import re
import operator
from math import log
import numpy as np

def PseKNC_code(seq):
    binary_dictionary={'A':[1,1,1],'T':[0,0,1],'G':[1,0,0],'C':[0,1,0],'N':[0,0,0]}
    nucleic_dictionary={'A':0,'T':0,'C':0,'G':0,'N':0}
    cnt=[]
    p=0
    for i in seq:
        temp=[]
        p=p+1
        nucleic_dictionary[i]+=1       
        temp=list(binary_dictionary[i])
        temp.append(nucleic_dictionary[i]/p)
        cnt.append(temp)
    return reduce(operator.add,cnt)
                  