# coding: utf-8
import os
from PIL import Image
from numpy import *
from random import shuffle
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
import numpy as np
import scipy.misc
from resizeimage import resizeimage
import warnings


from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
########################################################################################################################
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  inputs = tf.reshape(inputs, shape=[-1, 224, 224, 1])
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net

########################################################################################################################
n_classes = 40


Xtrain = []
Ytrain = []
Xtest = []
Ytest = []


import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 10
display_step = 0.001

# Network Parameters
n_input = 50176  # MNIST data input (img shape: 91*112)

dropout = 0.8  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 224, 224, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)



# Diviser notre base de donnée
def getTrainTest(n_classes):
    training_data = []
    for i in range(1, n_classes + 1):
        cible = zeros(n_classes)
        cible[i - 1] = 1
        X = []
        Y = []
        numberArrayExamples = open('ImageTest.txt', 'a')
        # Exemple = open('Xtest.txt', 'a')
        path = "att_faces/s" + str(i) + "/"

        for j in range(1, 11):
            a = zeros((224, 224))
            ch = path + str(j) + ".pgm"
            #ei  = Image.open(ch)
            #a[:, 9:101] = ei
            with Image.open(ch) as image:

                im = np.array(image)
                #print(im)
                #print(shape(im))
                a[56:168,46:138]=im

                imge = Image.fromarray(a)

                #cover = resizeimage.resize_cover(imge, [28, 28])
                covers = array(imge).reshape(224,224,1)
                im = np.array(covers)


                #eiar1 = str(im.tolist())
                #lineToWrite = "s" + str(i) + "/" + str(j) + '::' + eiar1 + '\n'
                #numberArrayExamples.write(lineToWrite)

                # im=misc.imread(ch)
                X.append(im)
                Y.append(cible)

            # partitionner la classe
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.5, random_state=40)
        # construire la base
        '''XtrText = str(Xtr)
        lineToWrite =  XtrText + '\n'
        Exemple.write(lineToWrite)'''

        Xtrain.extend(Xtr)
        Ytrain.extend(Ytr)
        Xtest.extend(Xte)
        Ytest.extend(Yte)
        training_data.append([Xtr, Ytr])

    return Xtest, Xtrain, Ytest, Ytrain

def test(x) :
    getTrainTest(n_classes)
    #shape = [1,28,28]
    #model = VGG16(include_top=False,weights='imagenet')
    pred = vgg_16(x,
           num_classes=n_classes,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    hm_epochs = 3
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())﻿
        sess.run(tf.initialize_all_variables())
        # Training our network
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(Xtrain):
                start = i
                end = i + batch_size
                # epoch_x, epoch_y = Dtrain.train.next_batch(batch_size)

                epoch_x = np.array(Xtrain[start:end])
                epoch_y = np.array(Ytrain[start:end])
                #print(epoch_x.shape)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 1.})
                epoch_loss += c
                i = i + batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss :', epoch_loss)

        # correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        # accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        # print('Accuracy',accuracy.eval({x:Xtest, y :Ytest}))
        #  Evaluate model
        '''correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #print('Accuracy', accuracy.eval({x: Xtest, y: Ytest, keep_prob: 1.}))
        batch_num = int(40/batch_size)
        test_accuracy = 0
        for i in range (batch_num) :
            
            test_accuracy += accuracy.eval({x: Xtest,
                                              y: Ytest,
                                              keep_prob: 1.0})
            test_accuracy /= batch_num
            print('test accuracy %g'%test_accuracy)'''


test(x)
