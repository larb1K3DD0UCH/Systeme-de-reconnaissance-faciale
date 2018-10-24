import os
import os.path as path
from convnetskeras.customlayers import crosschannelnormalization
from convnetskeras.customlayers import Softmax4D
from convnetskeras.customlayers import splittensor
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, Dense, Dropout, Flatten, Activation, merge
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from numpy import *
from random import shuffle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import scipy.misc
from resizeimage import resizeimage

n_classes = 40
MODEL_NAME = 'facerecognition_alexnet'
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
learning_rate = 0.001
training_iters = 300000
batch_size = 13
display_step = 0.001
n_input = 784
dropout = 0.25
def model_input(input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y = tf.placeholder(tf.float32, shape=[None, 40])
    return x, keep_prob, y

#x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input")
#y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32,name = "keep_prob")  # dropout (keep probability)

def getTrainTest(n_classes):
    training_data = []
    for i in range(1, n_classes + 1):
        cible = zeros(n_classes)
        cible[i - 1] = 1
        X = []
        Y = []
        path = "C:/Users/Mohammed/Desktop/att_faces/s" + str(i) + "/"

        for j in range(1, 11):
            a = zeros((112, 112))
            ch = path + str(j) + ".pgm"
            with Image.open(ch) as image:
                im = np.array(image)
                a[:, 9:101] = im

                imge = Image.fromarray(a)

                cover = resizeimage.resize_cover(imge, [28, 28])
                covers = array(cover).reshape(28, 28, 1)
                im = np.array(covers)
                X.append(im)
                Y.append(cible)

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=40)

        Xtrain.extend(Xtr)
        Ytrain.extend(Ytr)
        Xtest.extend(Xte)
        Ytest.extend(Yte)
        training_data.append([Xtr, Ytr])

    return Xtest, Xtrain, Ytest, Ytrain

def alex_net(x, keep_prob, y, output_node_name):


    _X = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(_X, 64, 3, 1, 'same', activation=tf.nn.relu)
    # 28*28*64
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')
    # 14*14*64

    conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)
    # 14*14*128
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')
    # 7*7*128

    conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)
    # 7*7*256
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'same')
    # 4*4*256

    flatten = tf.reshape(pool3, [-1, 4 * 4 * 256])
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    dropout = tf.nn.dropout(fc, keep_prob)
    logits = tf.layers.dense(dropout, 40)
    outputs = tf.nn.softmax(logits, name=output_node_name)

    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op

def train(x, keep_prob, y, train_step, loss, accuracy,merged_summary_op, saver):
    print("training start...")

    getTrainTest(n_classes)


    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.write_graph(sess.graph_def, 'out',MODEL_NAME + '.pbtxt', True)
        hm_epochs = 200
        summary_writer = tf.summary.FileWriter('logs/',graph=tf.get_default_graph())
        for epoch in range(hm_epochs):
            i = 0
            while i < len(Xtrain):
                start = i
                end = i + batch_size
                epoch_x = np.array(Xtrain[start:end])
                epoch_y = np.array(Ytrain[start:end])
                train_accuracy = accuracy.eval(feed_dict={x: epoch_x, y: epoch_y, keep_prob: 1.0})
                print('step %d, training accuracy %f' % (epoch, train_accuracy))
                _, summary = sess.run([train_step, merged_summary_op],feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})
                summary_writer.add_summary(summary, epoch)
                i = i + batch_size
        saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
        test_accuracy = accuracy.eval(feed_dict={x: Xtest,y: Ytest,keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")
def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names, [output_node_name],tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def main():

    if not path.exists('out'):
        os.mkdir('out')
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'
    x, keep_prob, y = model_input(input_node_name, keep_prob_node_name)
    train_step, loss, accuracy, merged_summary_op = alex_net(x, keep_prob,y, output_node_name)
    saver = tf.train.Saver()
    train(x, keep_prob, y, train_step, loss, accuracy,merged_summary_op, saver)
    export_model([input_node_name, keep_prob_node_name], output_node_name)


if __name__ == '__main__':
    main()