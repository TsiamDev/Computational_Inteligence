# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:57:59 2022
@author: Konstantinos Tsiamitros
"""

import re

def read_file(filename):
    #global storages
    glob_words = []
    glob_sent_num = []
    glob_word_num = []
    
    with open(filename) as f:
        lines = f.readlines()
        
        for line in lines:
            line2 = re.split(" |\< |\>", line)
            
            # storage for the enconded words
            words = []
            # storage for the number of sentences per line
            sent_num = []
            # storage for number of words in sentence i 
            word_num = []
            
            #print(line2)
            
            # get num of sentences
            temp = line2[0]
            temp = temp.split("<")
            temp = temp[1]
            #print(temp)
            sent_num.append(temp)
            
            # get num of words in sentence i
    
            temp = line2[2]
            temp = temp.split("<")
            temp = temp[1]
            #print(temp)
            word_num.append(temp)
    
            # temporary storage
            temp = []
            for x in line2[4:]:
                if x == "":
                    continue
                else:
                    try:
                        integer = int(x)
                        #print(integer)
                        temp.append(integer)
                    except Exception as ex:
                        #print(ex)
                        # New sentence encountered - store the new word_num
                        x = x.split("<")
                        #print(x[1])
                        word_num.append(x[1])
                        words.append(temp)
                        temp = []
            # do not forget about the last sentence of the last line!
            words.append(temp)
        
            glob_words.append(words)
            glob_sent_num.append(sent_num)
            glob_word_num.append(word_num)
    
    return glob_words, glob_sent_num, glob_word_num