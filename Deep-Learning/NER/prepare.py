# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:47:44 2017

@author: cm
"""

import os
import numpy as np
import pandas as pd
from hyperparameters import Hyperparamters as hp
import string


pwd = os.path.dirname(os.path.abspath('__file__'))
#print(pwd)



def load_vocabulary():
    word_dic = pd.read_csv('./dict/word_dict.txt', sep='\t', header=None)
    word = list(word_dic[0])
    return word

vocabulary = load_vocabulary()

def engWordOfSentence(sentence):
    letters = string.ascii_letters
    L = []
    length = len(sentence)
    engWord = ''
    i = 0
    while i < length:
        if sentence[i] in letters:
            engWord += sentence[i]
            if (i+1<length and sentence[i+1] not in letters) or i+1==length:
                L.append(engWord.capitalize())
                engWord = ''
        else:
            L.append(sentence[i])
        i += 1
    return L

def build_word_id(sentence):
#    L = [l for l in sentence]       # 需要修改成将 英文单词 部分当做一个整体
    L = engWordOfSentence(sentence)
    Id = []
    eos_index = vocabulary.index('<EOS>')
    unknow_index = vocabulary.index('unknow')
    for l in L:
        try:
            vocabulary.index(l)
            Id.append(vocabulary.index(l))
        except:
            Id.append(unknow_index)
    if len(Id) >= hp.num_steps:
        Id = Id[0:hp.num_steps]
    else: 
        Id.extend([eos_index] * (hp.num_steps - len(Id)))  
    return np.array(Id)


def build_id(sent):
    LS = sent.split()
    words_id = []
    labels_id = []
    for i,L in enumerate(LS):
        ls = L.split('/')
        if len(ls)==2:
            word = ls[0]
            label = ls[1]
            word_list = [w for w in word]
            for w in word_list:
                try:
                    vocabulary.index(w)
                    words_id.append(vocabulary.index(w))
                    if label == 'ns' or label == 'nsf':
                        if word_list.index(w) == 0:
                            labels_id.append(1)  # B 1
                        else:
                            labels_id.append(2)  # I 2
                    else:
                        labels_id.append(0)      # O 0
                except:
                    words_id = words_id
                    labels_id = labels_id
        else:
            continue
    return words_id,labels_id



def build_id_n(sentences):
    w_ids = []
    l_ids = []
    for i in range(len(sentences)):        
        w_id,l_id = build_id(sentences[i])
        if len(l_id) >= hp.num_steps:
            w_ids.append(w_id[0:hp.num_steps])
            l_ids.append(l_id[0:hp.num_steps])
        elif len(l_id) < hp.num_steps:
            l = len(l_id)    
            w_id.extend([0]*(hp.num_steps-l))
            w_ids.append(w_id)
            l_id.extend([0]*(hp.num_steps-l))
            l_ids.append(l_id)           
    return np.array(w_ids),np.array(l_ids)


if __name__ == '__main__':
    text = '我爱/n 中国/ns 人/l  aa/d'
    text = '[武汉市/ns 江夏区/ns]/nz'
    build_id(text)
    a,b = build_id_n(['我爱/n 中国/ns ','我爱/n 武汉/ns'])
    a,b = build_id_n([text])
    print(a)
