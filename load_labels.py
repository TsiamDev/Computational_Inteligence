# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:55:41 2022

@author: Konstantinos Tsiamitros
"""

import numpy as np

def load_labels(filename):
    with open(filename) as f:
        lines = f.readlines()
        
        labels = []
        for line in lines:
            doc_i_labels = []
            for lb in line:
                if (lb is " ") or (lb is "\n"):
                    continue
                doc_i_labels.append(int(lb))
            labels.append(doc_i_labels)
        
        #print(labels)
        return np.array(labels)
        
#load_labels()