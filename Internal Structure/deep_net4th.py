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
import ascon_permutation1 as ac
from time import time
import matplotlib.pyplot as plt
from numba import cuda
from sklearn.metrics import confusion_matrix

bs = 1000
wdir = './freshly_trained_nets/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

def make_transformer(vocab_size = 2,sequence_length = 640, embed_dim = 64,num_heads = 2,dense_dim = 256):
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.summary()
    return(model)


def make_resnet(num_blocks=5, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=64, ks=3, depth=1, reg_param=0.0001, final_activation='sigmoid'):
  # Input and preprocessing layers
  inp = Input(shape=(2*num_blocks * word_size ,))
  rs = Reshape((2*num_blocks, word_size))(inp)
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
  model.summary()
  return(model)

def make_mlp(num_blocks=5, word_size=64,reg_param=10**-5):
  # Input and preprocessing layers

  model = keras.Sequential()
  model.add(Input(shape=(num_blocks * word_size*2,)))
  model.add(Dense(640, input_dim=640, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(640, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(640, activation='relu', kernel_regularizer=l2(10 ** -5)))
  model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10 ** -5)))
  model.summary()
  return(model)


def train_ac_distinguisher(num_epochs, num_rounds,S,diffs):
    # generate training and validation data
    X, Y = ac.make_dataset(n=3*10 ** 6, Nr=num_rounds,df=diffs)
    X_eval, Y_eval = ac.make_dataset(n=3*10 ** 5, Nr=num_rounds,df=diffs)
    X_test, Y_test = ac.make_dataset(n=3*10 ** 5, Nr=num_rounds, df=diffs)
    #X_real, Y_real = ac.make_dataset_diff_real(n=10 ** 5, Nr=num_rounds, df=diffs)
    #X_random, Y_random = ac.make_dataset_diff_random(n=10 ** 5, Nr=num_rounds,df=diffs)
    # 方法1：设置字体为系统支持的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    '''
    train_set = ac.make_dataset_mc_diff(n=10 ** 7, Nr=num_rounds,s=S,df=diffs)
    print("222222222222222222222")
    np.save('107s4train_set4.npy',train_set)
    vali_set = ac.make_dataset_mc_diff(n=10 ** 6,Nr=num_rounds,s=S,df=diffs)
    np.save('107s4vali_set4.npy', vali_set)
    test_set = ac.make_dataset_mc_diff(n=10 ** 6, Nr=num_rounds,s=S,df=diffs)
    np.save('107s4test_set4.npy', test_set)
    print("1111111111111")
    #train_set = ac.make_dataset_mc_diff(n=10 ** 6, Nr=num_rounds, s=S, df=diffs)
    #np.savetxt(r'106s32train_set4.txt',train_set,delimiter=',')
    train_set_loaded_np = np.load('107s4train_set4.npy')
    #X = train_set_loaded_np[:, :10240]
    #Y = train_set_loaded_np[:, 10240:]
    #np.savetxt('106s32train_set4.txt',train_set_loaded_np,fmt='%s',newline='\n')
    vali_set_loaded_np = np.load('107s4vali_set4.npy')
    X_eval = vali_set_loaded_np[:, :1280]
    Y_eval = vali_set_loaded_np[:, 1280:]
    test_set_loaded_np = np.load('107s4test_set4.npy')
    X_test = test_set_loaded_np[:, :1280]
    Y_test = test_set_loaded_np[:, 1280:]


    def data_generator(matrix,batch_size):
        """自定义数据生成器"""
        while True:
            cnt = 0
            X = []
            Y = []
            for start in range(0, matrix.shape[0]):
                x = np.array(matrix[start,:1280])
                y = np.array(matrix[start,1280:])
                X.append(x)
                Y.append(y)
                cnt +=1
                if cnt == batch_size:
                    cnt = 0
                    yield(np.array(X),np.array(Y))
                    X=[]
                    Y=[]
    '''
    #create the network
    t0=time()
    #net = make_mlp(num_blocks=5, word_size=64,reg_param=10**-5)
    #net = make_resnet(num_blocks=5, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=64, ks=3, depth=1, reg_param=0.0001, final_activation='sigmoid')
    net = make_transformer(vocab_size=2, sequence_length=640, embed_dim=64, num_heads=2, dense_dim=256)
    #net = make_transformer(vocab_size=1, sequence_length=640, embed_dim=64, num_heads=1, dense_dim=320)
    #net = make_transformer(vocab_size = 1, sequence_length = 640, embed_dim = 64, num_heads = 1, dense_dim = 32)
    #net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']);
    net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']);
    #set up model checkpoint
    earlypasuse =  EarlyStopping(monitor="val_accuracy",patience =7,min_delta=0.0001,mode='max',restore_best_weights=True,verbose=1)
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(50, 0.002, 0.0001));
    #lr = ReduceLROnPlateau(monitor='val_acc',factor=0.1,patience=4,min_lr=0.0001)
    tensorboard = TensorBoard(log_dir = wdir)
    callbacks_list = [earlypasuse,check,lr,tensorboard]
    #train and evaluate
    #h = net.fit(data_generator(train_set_loaded_np,bs),steps_per_epoch=100, epochs=num_epochs, validation_data=(X_eval, Y_eval), callbacks=callbacks_list)
    #h = net.fit(generate_arrays_from_file('106s32train_set4.txt', batch_size=bs),steps_per_epoch=100, epochs=num_epochs, validation_data=(X_eval, Y_eval), callbacks=callbacks_list)
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data = (X_eval, Y_eval),callbacks=callbacks_list);
    #h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
    #            validation_split = 0.2,callbacks=callbacks_list);
    #print(earlypasuse)
    t1=time()
    bestval = np.max(h.history['val_accuracy'])
    t=(t1-t0)
    print("Best validation accuracy: ", np.max(h.history['val_accuracy']));

    #model = keras.models.load_model(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5',custom_objects={'PositionalEmbedding':PositionalEmbedding})
    custom_objects = {'PositionalEmbedding':PositionalEmbedding,'TransformerEncoder':TransformerEncoder}
    model = tf.keras.models.load_model(wdir + 'best' + str(num_rounds) + '.h5',custom_objects=custom_objects)


    Y_prep=model.predict(X_test)
    y_pred = np.apply_along_axis(lambda x: 1 if x > 0.5 else 0,1,Y_prep)
    print(y_pred)
    y_true = Y_test.reshape((300000,))
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
    test_score = model.evaluate(X_test,Y_test)

    # 打印出来
    actual_epochs = len(h.history['loss'])
    history_dict = h.history
    fig = plt.figure()
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, actual_epochs+1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(epochs, loss_values, "r.-", label="Training loss")
    ax1.plot(epochs, val_loss_values, "b.-", label="Validation loss")
    ax1.set_title("Training and validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2 = fig.add_subplot(2, 1, 1)
    ax2.plot(epochs, acc, "r.-", label="Training acc")
    ax2.plot(epochs, val_acc, "b.-", label="Validation acc")
    ax2.set_title("Training and validation accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(wdir+'best'+str(num_rounds)+'.eps',dpi=300,transparent=True,bbox_inches="tight",pad_inches=0.0)
    plt.savefig(wdir + 'best' + str(num_rounds) + '.jpeg', dpi=300)
    f = open(r"result.txt","a")
    f.write("num_round="+str(num_rounds)+":"+"\n")
    f.write("Best validation accuracy: "+str(bestval)+"\n")
    f.write("test accuracy: " + str(test_score[1]) + "\n")
    f.write("confusion_matrix: " + str(cm) + "\n")
    f.write("wall time per attack(in minutes):" + str(t) + "s" + "\n")
    f.write("epoch: " + str(actual_epochs) + "\n")
    f.close()
    #plt.show()
    # save model
    return(net, h)
    cuda.select_device(0)
    cuda.close()


if __name__ == '__main__':
    '''
    for i in range(1,2,1):
        train_ac_distinguisher(num_epochs=50, num_rounds=i)
    '''
    #my_array = [0x0000000000000001,0x0000000000000100,0x0000000000010000,0x0000000001000000,0x0000000100000000,0x0000010000000000,0x0001000000000000,0x0100000000000000,0x8000000000000000]
    my_array = [0x0000000000000001]
    #my_array =[0x0000100000000001,0x0040000000000001,0x8000000000000001,0x0000000010000001,0x0000004000000001,0x0040800000000001,0x0000000010080001,0x8040000000040000,0x0000100040000001,0x0000100004040000,0x8040800000000001,0x1040800000000001,0x0040800000010001,0x0040800001000001,0x0040800100000001,0x8000100801000004,0x022030304201080d,0xc0489800262500a0,0x04314f4725f80001,0x441326c0236cca84,0x04364206f5a80802,0xc01d986058edb14f,0x04364206f5a80802]
    for value in my_array:
        train_ac_distinguisher(num_epochs=50, num_rounds=2,S=16,diffs=value)
        print(value)
