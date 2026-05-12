import numpy as np
from pickle import dump
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import ascon_run as ac
from time import time
import matplotlib.pyplot as plt
from numba import cuda
from sklearn.metrics import confusion_matrix


bs = 10000
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);


def make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=64, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
  # Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,))
  rs = Reshape((2 * num_blocks, word_size))(inp)
  perm = Permute((2, 1))(rs)
  # add a single residual layer that will expand the data to num_filters channels
  # this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  conv0 = BatchNormalization()(conv0)

  conv0 = Activation('relu')(conv0)
  # add residual blocks
  shortcut = conv0
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    shortcut = Add()([shortcut, conv2])
  # add prediction head
  flat1 = Flatten()(shortcut)
  dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)
  return(model)

def make_mlp(num_blocks=4, word_size=64,reg_param=10**-5):
  # Input and preprocessing layers

  model = keras.Sequential()
  model.add(Input(shape=(num_blocks * word_size * 2,)))
  model.add(Dense(512, input_dim=512, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(512, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(512, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10 ** -5)))
  return(model)






def train_ac_distinguisher(num_epochs, num_rounds, depth=1):

    #generate training and validation data
    X, Y = ac.make_dataset(n=10**6,Nr=num_rounds)
    X_eval, Y_eval = ac.make_dataset(n=10**5, Nr=num_rounds)
    X_test, Y_test = ac.make_dataset(n=10 ** 5, Nr=num_rounds)
    # create the network
    t0 = time()
    #net = make_mlp(num_blocks=4, word_size=64, reg_param=10 ** -5)
    net = make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=64, ks=3, depth=1,
                      reg_param=0.0001, final_activation='sigmoid')
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']);

    earlypasuse = EarlyStopping(monitor="val_accuracy", patience=7, min_delta=0.0001, mode='max',
                                restore_best_weights=True, verbose=1)
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + '.h5');
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(50, 0.002, 0.0001));
    # lr = ReduceLROnPlateau(monitor='val_acc',factor=0.1,patience=4,min_lr=0.0001)
    tensorboard = TensorBoard(log_dir=wdir)
    callbacks_list = [earlypasuse, check, lr, tensorboard]

    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data=(X_eval, Y_eval), callbacks=callbacks_list);
    t1=time()
    bestval = np.max(h.history['val_accuracy'])
    t=(t1-t0)/60
    print("Best validation accuracy: ", np.max(h.history['val_accuracy']))
    print("wall time per attack(in minutes):",(t1-t0)/60)
    actual_epochs = len(h.history['loss'])
    Y_prep = net.predict(X_test)
    y_pred = np.apply_along_axis(lambda x: 1 if x > 0.5 else 0, 1, Y_prep)
    print(y_pred)
    y_true = Y_test.reshape((100000,))
    print(y_true.shape)
    print(y_pred.shape)
    cm = confusion_matrix(y_true, y_pred)
    print(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    FPR = fp / (tn + fp)
    FNR = fn / (fn + tp)
    print("混淆矩阵：\n", cm)
    print("真阳率TPR=" + str(TPR) + "\n")
    print("真阴率TNR=" + str(TNR) + "\n")
    print("假阳率FPR=" + str(FPR) + "\n")
    print("假阴率FNR=" + str(FNR) + "\n")
    print("wall time per attack(in s):", t)
    test_score = net.evaluate(X_test, Y_test)
    f = open(r"result.txt", "a")
    f.write("num_round=" + str(num_rounds) + ":" + "\n")
    f.write("Best validation accuracy: " + str(bestval) + "\n")
    f.write("test accuracy: " + str(test_score[1]) + "\n")
    f.write("confusion_matrix: " + str(cm) + "\n")
    f.write("wall time per attack(in minutes):" + str(t) + "s" + "\n")
    f.write("epoch: " + str(actual_epochs) + "\n")
    f.close()
    # plt.show()
    # save model
    return (net, h)
    cuda.select_device(0)
    cuda.close()


if __name__ == '__main__':
    for i in range(3,12,1):
        train_ac_distinguisher(num_epochs=50, num_rounds=i)
