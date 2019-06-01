# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
#from prepare import build_id_n
from networks  import BiLSTM_CFR
from hyperparameters import Hyperparamters as hp
import pandas as pd


pwd = os.path.dirname(os.path.abspath('__file__'))

print(1)
# loading model
BiRNN_model = BiLSTM_CFR(num_steps = hp.num_steps,
                        n_hidden = hp.n_hidden,
                        batch_size = hp.batch_size,
                        n_class = hp.n_classes ,
                        vocab_size = hp.vocab_size,
                        learning_rate = hp.learning_rate)
                
#print(1)

def shuffle_one(a1):
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    return a1[ran]


def select(data,ids):
    data = [data[i] for i in ids]
    data = [[eval(p) for p in pair] for pair in data]
    X = np.array(data)[:,0]
    Y = np.array(data)[:,1]
    return X, Y

def accuracy(y,ypred): # y:[batch_size]*[num_steps]
    n,p = y.shape
    N,r=0,0
    for i in range(n):
        for j in range(p):
            if y[i,j] in [1,2]:
                N = N+1
                if y[i,j] == ypred[i,j]:
                    r = r+1
                else:
                    r = r
            else:
                N = N
                r = r
    return r/N


# loading data
data = pd.read_csv('./data/ner_200000.csv', skiprows=1, encoding = 'utf-8',names=['X','Y'],header=None)
data = np.array(data)


# traning data--test data
N = len(data)
N_train = np.int(N*hp.train_rate)
N_test = N - N_train
data_train =  data[0:N_train] 
data_test =  data[N_train:N] 
b = np.arange(N_train)               
n_batches = int((N_train - 1) /hp.batch_size)
print('number of batch:',n_batches)
del data   
      

#print(3)


# start graph training
#saver = tf.train.Saver(max_to_keep=2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


## # restore parameters
#MODEL_SAVE_PATH = os.path.join(pwd,'model')
#ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
#if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print('Restored model!')

with sess.as_default():
    summary_writer = tf.summary.FileWriter('./model', graph=sess.graph)
    for i in range(hp.n_epoch):
        
        b = shuffle_one(b)  # randomization for all training data in every epoch
        for batch_num in range(n_batches-1):
            i1 = b[batch_num * hp.batch_size:min((batch_num + 1) * hp.batch_size, N_train)]
#            x3,y3 = build_id_n(select(data_train,i1))
            x3, y3 = select(data_train,i1)
            
            fd = {BiRNN_model.x: x3,
                  BiRNN_model.y: y3,
                  BiRNN_model.keep_prob:0.5}
            sess.run(BiRNN_model.optimizer, feed_dict = fd)

#            if batch_num%1000==0:
#                print (batch_num)

            if batch_num % hp.print_step == 0:
                
                fd2 = {BiRNN_model.x: x3,
                      BiRNN_model.y: y3,
                      BiRNN_model.keep_prob:1}
                
                tf_scores = sess.run(BiRNN_model.outputs,feed_dict=fd2)
                tf_transition_params = sess.run(BiRNN_model.transition_params,feed_dict=fd2)
                tf_scores = np.squeeze(tf_scores)
                pred = []
                for logit, seq_length in zip(tf_scores, [hp.num_steps]*hp.batch_size):
                    logit = logit[:seq_length]
                    vtb_seq, vtb_score = tf.contrib.crf.viterbi_decode(logit,tf_transition_params)
                    pred += [vtb_seq]

                
                pred = np.array(pred)

                [loss, summary] = sess.run([BiRNN_model.loss, BiRNN_model.summary_op], feed_dict = fd2)
                summary_writer.add_summary(summary, batch_num)

                accu = accuracy(y3,pred)

                time_str = time.strftime('%Y-%m-%d %X', time.localtime())
                print("{}: Iter {}, batch {:g}, acc {:g}, loss {:g}".format(time_str, i, batch_num, accu, loss))
                BiRNN_model.saver.save(sess, os.path.join(pwd, 'model', 'model.ckpt'), global_step=i)
    print('Optimization finished')








