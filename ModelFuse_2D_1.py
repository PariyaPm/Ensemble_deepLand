#s is an implementation of 3DCNN model of deepLandU

# Import all modules
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model
import tensorflow as tf
from keras.layers.merge import concatenate, add, average
from keras.layers import Activation, Dropout, Dense, Reshape
from keras.layers.convolutional import Conv3D,Conv3DTranspose
from keras.layers.convolutional import MaxPooling3D
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras import layers
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import History
import os
from keras.utils import multi_gpu_model

from io import BytesIO
import numpy as np
import requests
import shutil
import time

#import theano

#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

#from tempfile import TemporaryFile

if K.backend()=='tensorflow':
    K.set_image_dim_ordering('th')

def auc_roc(y_true, y_pred):
#    y_pred = np.argmax(y_pred,axis=1)

    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true, curve='ROC')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    
    

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc



batch_size = 32
epochs = 200
 
code_path = '.'
data_path = '.'

os.chdir(code_path)


####Clean_up, pre-process the data and organize it as train and test data sets
#data_train = np.load( 'x_patches.npy')
#labels_train = np.load( 'y_patches.npy')
y = requests.get('https://data26913378jstar.s3-us-west-1.amazonaws.com/y_patches.npy',stream = True)
with open('datY.npy', 'wb') as fin:
	shutil.copyfileobj(y.raw, fin)
labels_train =  np.load('datY.npy')
print('yshape is ', labels_train.shape)
os.remove('datY.npy')

x = requests.get('https://xdataset.s3-us-west-1.amazonaws.com/x_patches.npy', stream = True)
with open('datX.npy', 'wb') as fin:
        shutil.copyfileobj(x.raw, fin)
data_train =  np.load('datX.npy')
os.remove('datX.npy')

#data_train =  np.load(BytesIO(x.raw.read()), allow_pickle = True)
print('xshape is' , data_train.shape)

el = requests.get('https://data26913378jstar.s3-us-west-1.amazonaws.com/elev.txt', stream = True)
elev =  np.genfromtxt(BytesIO(el.raw.read()), delimiter = ' ', skip_header = 6)
print('elev shape is' , elev.shape)

#add a dimension to the training data
labels_train = np.expand_dims(labels_train, axis=1)

print(labels_train.shape)

data_train = np.expand_dims(data_train, axis=4)
labels_train = np.expand_dims(labels_train, axis=4)
print(data_train.shape)
print(labels_train.shape)

#find the patches that are not in the study area
#elev = np.genfromtxt('./elev.txt', delimiter=' ', skip_header = 6)
elev_patches = np.array([elev[i:i + 32, j:j + 32] for i in range(0, elev.shape[0], int(32)) for j in range(0, elev.shape[1], int(32))])
exc_ind = []
for i in range(elev_patches.shape[0]): 
    if(np.sum(elev_patches[i])==-10238976.0):
       exc_ind.append(i)
       
elev = None
#generate random indices for the partitioned data
range_ind = np.arange(52455)
np.random.seed(1364)

range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)
x_train= data_train[range_ind[:int(len(range_ind)*.66)],:,:,:,:]
x_test = data_train[range_ind[int(len(range_ind)*.66):],:,:,:,:]
y_train = labels_train[range_ind[0:int(len(range_ind)*.66)],:,:,:,:]
y_test= labels_train[range_ind[int(len(range_ind)*.66):],:,:,:,:]

print(x_train.shape[1:])

model1 = Sequential()

model1.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', input_shape = x_train.shape[1:]))
model1.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Dropout(0.5))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Dropout(0.5))

model1.add(Conv2D(512, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(UpSampling2D(size=(2,2)))

model1.add(Conv2D(512, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(512, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(256, (2,2), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(UpSampling2D(size=(2,2)))

model1.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(UpSampling2D(size=(2,2)))

model1.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(UpSampling2D(size=(2,2)))

model1.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(2, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Conv2D(1, 1))


model2 = Sequential()

model2.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', input_shape = x_train.shape[1:], data_format= 'channels_first'))
model2.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Dropout(0.10))
model2.add(AveragePooling2D(pool_size=(2, 2), data_format= 'channels_first'))

model2.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Dropout(0.10))
model2.add(MaxPooling2D(pool_size=(2, 2), data_format= 'channels_first'))

model2.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(512, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(1024, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Dropout(0.10))
model2.add(MaxPooling2D(pool_size=(2, 2), data_format= 'channels_first'))

model2.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(256, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Dropout(0.10))
model2.add(UpSampling2D(size=(2,2), data_format= 'channels_first'))

model2.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(128, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Dropout(0.10))
model2.add(UpSampling2D(size=(2,2), data_format= 'channels_first'))

model2.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))  
model2.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(64, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
model2.add(Conv2D(2, (3,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal', data_format= 'channels_first'))
#    model.add(Conv2D(1, 1, activation = 'sigmoid'))
model2.add(Dropout(0.10))
model2.add(UpSampling2D(size=(2, 2),data_format= 'channels_first'))
#    model.add(Dense(32, input_dim = (1,32,32)))
model2.add(Conv2D(1, 1, data_format= 'channels_first'))
#model2.add(Activation('sigmoid'))
print(model2.output.shape)

model_concat = add([model1.output, model2.output])
#model_concat = Dense(1 , activation='sigmoid')(model_concat)
#model_concat = AveragePooling2D(size=(1, 2))
print(model_concat.shape)
#model_concat = (Conv3DTranspose(2, (6,1,1), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last'))
model_concat = Dropout(0.1)(model_concat)
#print(model_concat.output.shape)
model_concat = Activation('sigmoid')(model_concat)

model = Model(inputs=[model1.input, model2.input], outputs=model_concat)

deepLandF = multi_gpu_model(model, gpus=4)
print(deepLandF.summary())

print('blah')

 # Train model
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
deepLandF.compile(optimizer=Adam(lr = 0.0002),
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc_roc, dice_coef] )

print(deepLandF.summary())

print('blah')

import time

start_time = time.time()
print('5')

# Fit model
cnn = deepLandF.fit([x_train,x_train], y_train,
                batch_size= batch_size, epochs=epochs,
                validation_data=([x_test, x_test],y_test), shuffle=False)


y_predict = deepLandF.predict([x_test, x_test], batch_size= batch_size)

print('6')

end_time = time.time()  

print("the running time for fit predict is", end_time-start_time)
#cnn = deepLandF.fit(x_train, y_train,
#               batch_size= batch_size, epochs=epochs,
#               validation_data=(x_test,y_test), shuffle=True)
y_class = y_predict
y_class[y_class>=0.5] = 1
y_class[y_class<0.5] = 0

y_predict = deepLandF.predict([x_test, x_test], batch_size= batch_size)

np.save("y_predictdeepLandF2",y_predict)
np.save("y_classdeepLandF2",y_class)

#load cnnmodel and do predict and class once again

w = deepLandF.get_weights()
np.save('weightsdeepLandF2', w)

histdict = cnn.history

import csv 

  
with open("deepLandF2.csv", "w", newline = '') as file:
    w = csv.writer(file)
    for key, val in histdict.items():
        print(key, val)
        w.writerow([key, val])
file.close()
