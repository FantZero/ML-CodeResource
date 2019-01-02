# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:58:47 2018

@author: jz
"""
import tensorflow as tf
import numpy as np

# store variables
#W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
#b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#
#init = tf.initialize_all_variables()
#
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, './my_variable/test.ckpt')

# restore variables
# redefine the same shape and same type for your variables
W2 = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b2= tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')
# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./my_variable/test.ckpt")
    print (sess.run(W2))
    print (sess.run(b2))



