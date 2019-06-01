# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:46:29 2017

@author: cm
"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from networks import BiLSTM_CFR
from prepare import build_word_id
#from utils import cut_sentence
from hyperparameters import Hyperparamters as hp


tag_key = ['person_name', 'org_name', 'location', '<EOS>']
tag_value = ['pn', 'on', 'loc', 'EOS']
tag_dict = dict(zip(tag_value, tag_key))

pwd = os.path.dirname(os.path.abspath(__file__))

class model_BiLSTM_CRF(object,):
    """
    加载 BiRNN_crf 神经网络模型
    """
    def __init__(self):
        self.bilstm_crf, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            checkpoint_dir = os.path.join(pwd, "model")
            with sess.as_default():
                bilstm_crf =  BiLSTM_CFR(num_steps=hp.num_steps,
                                   n_hidden=hp.n_hidden,
                                   batch_size=1,
                                   n_class=hp.n_classes,
                                   vocab_size=hp.vocab_size,
                                   learning_rate=hp.learning_rate)
                sess.run(tf.global_variables_initializer())
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                bilstm_crf.saver.restore(sess, ckpt.model_checkpoint_path)
        return bilstm_crf,sess

model = model_BiLSTM_CRF()


def prediction(sentence):
    """
    预测一句话中的中文
    """
    index = build_word_id(sentence)
    x = index.reshape((1,hp.num_steps))
    fd = { model.bilstm_crf.x:x,
           model.bilstm_crf.keep_prob: 1.0}
    tf_scores = model.sess.run(model.bilstm_crf.outputs, 
                               feed_dict=fd)
    tf_transition_params = model.sess.run(model.bilstm_crf.transition_params, 
                                          feed_dict=fd)
    tf_scores = np.squeeze(tf_scores)
    logit =  tf_scores[:hp.num_steps]
    vtb_seq, vtb_score = tf.contrib.crf.viterbi_decode(logit, tf_transition_params)
    return vtb_seq
    
class local_bilstm_crf(object):
    """
    取出句子中的地点名词
    """
    def __init__(self, sentence, simpleVersion=False):
        
        self.sentence = sentence
        self.simpleVersion = simpleVersion
        
        def getDict(dictFile):
            word_dic = pd.read_csv(dictFile, sep='\t', header=None)
            data = list(word_dic[0])
            data_id = list(word_dic[1])
            data2id = dict(zip(data, data_id))
            id2data = dict(zip(data_id, data))
            return data2id, id2data

        def prediction_one(sentence):
            _, id2tag = getDict('./dict/tag_dict.txt')
            res = []
            prd = list(prediction(sentence))
            prd = prd[:len(sentence)]
            for i in range(len(prd)):
                res.append(id2tag[prd[i]])
            return res#,res,inds


        def prediction_n(sentence):
            '''
                根据 simpleVersion 参数 选择是否返回简易版本
                完整版本：[{"sentence": "李磊在武汉市黄鹤楼游玩", "entity": [[0, 1, "person_name"], [3, 8, "location"]]}]
                简易版本：{'location':['北京市朝阳区', '武汉市黄鹤楼'], 'person_name':['李磊', '李白']}
            '''
            locations = prediction_one(sentence)
            full_result = {}
            simple_result = {}
            if len(sentence) == len(locations):
                i = 0
                entity = []
                tagSet = set()
                while i < len(locations):
                    if 'B' in locations[i]:
                        tag = locations[i].split('_')[1]
                        j = i+1
                        while j < len(locations):
                            if 'I' in locations[j]:
                                if j+1<len(locations) and ('B' in locations[j+1] or '/O'==locations[j+1]):
                                    break
                                else:
                                    j += 1
                            else:
                                break
                        tagSet.add(tag_dict[tag])
                        entity.append([i, j, tag_dict[tag]])
                        i = j
                    else:
                        i += 1
                full_result['sentence'] = sentence
                full_result['entity'] = entity
                
                for t in tagSet:
                    simple_result[t] = [sentence[ent[0]:ent[1]+1] for ent in entity if ent[2]==t]
            if self.simpleVersion :
                return full_result, simple_result 
            else:
                return full_result

        self.prediction = prediction_n(self.sentence)


if __name__ == '__main__':
    sent = '我是张艺凡，我在北京市朝阳区人民法院的门口,北京市朝阳区人民法院,巴黎'
    sent = '在中国这块土地上，有钱学森这样的物理学家、也有鲁迅、莫言这样的文学家，在中国共产党的领导下共进退。'
    print(local_bilstm_crf(sent, True).prediction)
    full_result, simple_result = local_bilstm_crf(sent, True).prediction
    texts = [
     '北京市朝阳区人民法院',
     '我在武汉和北京以及南昌工作过',
     '哈哈哈哈哈哈',
     '饶亚庆是武汉人',
     '中兴',
     '陈伟是福建人',
     '陈伟是武汉人',
     '山东半岛',
     '山东半岛濒临黄渤海',
     '刘德华今天去了广州的微信研发中心',
     '我住在武汉市江夏区',
     '经柳林大道、银杏大街 、四里河路 、西一环路 、清溪路',
     '未来科技城',
     '中国共产党',
     '我是张艺凡，我在北京市朝阳区人民法院的门口,北京市朝阳区人民法院,法国巴黎',
     '在中国这块土地上，有钱学森这样的物理学家、也有鲁迅、莫言这样的文学家，在中国共产党的领导下共进退。'
     ]
    res = []
    for text in texts:
        text = text.replace(' ','')
        res_sent = local_bilstm_crf(text, True).prediction
        res.append(res_sent)
        print(res_sent)






