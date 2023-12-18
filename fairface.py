import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, Add, Input
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical, plot_model
import cv2
import keras.callbacks
import os
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder


class Logger(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gender_accuracy = logs.get('gender_accuracy')
    race_accuracy = logs.get('race_accuracy')
    age_accuracy = logs.get('age_accuracy')
    val_gender_accuracy = logs.get('val_gender_accuracy')
    val_race_accuracy = logs.get('val_race_accuracy')
    val_age_accuracy = logs.get('val_age_accuracy')
    print('='*30, epoch + 1, '='*30)
    print('gender_accuracy: {}, race_accuracy: {}, age_accuracy : {}'.format(gender_accuracy, race_accuracy, age_accuracy))
    print('val_gender_accuracy: {}, val_race_accuracy: {}, val_age_accuracy : {}'.format(val_gender_accuracy,val_race_accuracy, val_age_accuracy))



#setting number of images to be used
num_train = 200
num_val = 100
#Data loading and preprossing
train_label = pd.read_csv('fairface_label_train.csv')
val_label = pd.read_csv('fairface_label_val.csv')

train_path = './fairface-img-margin025-trainval/train'
val_path = './fairface-img-margin025-trainval/val'

train_list = os.listdir(train_path)
val_list = os.listdir(val_path)

x_train =[]
x_val = []

for elem in train_list:
    path = train_path + '/'+ str(elem)
    im = cv2.imread(path)
    x_train.append(im)
    if len(x_train) == num_train:
        break

for elem in val_list:
    path = val_path + '/'+str(elem)
    im = cv2.imread(path)
    x_val.append(im)
    if len(x_val) == num_val:
        break


x_train = np.asarray(x_train)
x_val = np.asarray(x_val)
x_train = x_train/255.
x_val = x_val/255.


label_encoder = LabelEncoder()

train_label_age = train_label['age']
train_label_age_encoded = label_encoder.fit_transform(train_label_age)
train_label_age = np.asarray(train_label_age_encoded)

train_label_gender = train_label['gender']
train_label_gender = np.asarray(train_label_gender)

train_label_race = train_label['race']
train_label_race_encoded = label_encoder.fit_transform(train_label_race)
train_label_race = np.asarray(train_label_race_encoded)

val_label_age = val_label['age']
val_label_age_encoded = label_encoder.fit_transform(val_label_age)
val_label_age = np.asarray(val_label_age_encoded)

val_label_gender = val_label['gender']
val_label_gender = np.asarray(val_label_gender)

val_label_race = val_label['race']
val_label_race_encoded = label_encoder.fit_transform(val_label_race)
val_label_race = np.asarray(val_label_race_encoded)

btrain_label_gender = (train_label_gender == 'Male').astype(int)
bval_label_gender = (val_label_gender == 'Male').astype(int)

train_label_age = to_categorical(train_label_age)
train_label_race = to_categorical(train_label_race)

val_label_age = to_categorical(val_label_age)
val_label_race = to_categorical(val_label_race)

#model building
input_ = Input(shape=(224,224,3), name = 'input_')
pre_model = ResNet50(include_top= False, )(input_)


conv_1 = Conv2D(32,3,name = 'conv_1')(pre_model)
act_1 = Activation('relu', name = 'act_1')(conv_1)

conv_2 = Conv2D(32,3, name = 'conv_2')(act_1)
act_2 = Activation('relu', name = 'act_2')(conv_2)
pool_1 = MaxPool2D(2, name = 'pool_1')(act_2)
flat_1 = Flatten(name = 'flat_1')(pool_1)
dense_1 = Dense(100,activation='relu',name = 'dense_1')(flat_1)
dense_2 = Dense(10,activation='relu',name = 'dense_2')(dense_1)
gender = Dense(1, activation='sigmoid', name = 'gender')(dense_2)

conv_3 = Conv2D(32,3,padding = 'same',name = 'conv_3')(pre_model)
act_3 = Activation('relu', name = 'act_3')(conv_3)
conv_4 = Conv2D(32,3,padding = 'same',name = 'conv_4')(act_3)
act_4 = Activation('relu', name = 'act_4')(conv_4)
conv_5 = Conv2D(32,3, padding = 'same', name = 'conv_5')(act_4)
add_1 = Add(name = 'add_1')([conv_5,act_3])
act_5 = Activation('relu',name ='act_5')(add_1)
pool_2 = MaxPool2D(2,name = 'pool_2')(add_1)
flat_2 = Flatten(name = 'flat_2')(pool_2)
dense_3 = Dense(100,activation='relu',name = 'dense_3')(flat_2)
dense_4 = Dense(50,activation='relu', name = 'dense_4')(dense_3)
race = Dense(7,activation='softmax', name = 'race')(dense_4)


conv_6 = Conv2D(32,3,padding = 'same',name = 'conv_6')(pre_model)
act_6 = Activation('relu', name = 'act_6')(conv_6)
conv_7 = Conv2D(32,3,padding = 'same',name = 'conv_7')(act_6)
act_7 = Activation('relu', name = 'act_7')(conv_7)
conv_8 = Conv2D(32,3, padding = 'same', name = 'conv_8')(act_7)
add_2 = Add(name = 'add_2')([conv_8,act_6])
act_8 = Activation('relu',name ='act_8')(add_2)
pool_3 = MaxPool2D(2,name = 'pool_3')(add_2)
flat_3 = Flatten(name = 'flat_3')(pool_3)
dense_5 = Dense(100,activation='relu',name = 'dense_5')(flat_3)
dense_6 = Dense(50,activation='relu', name = 'dense_6')(dense_5)
age = Dense(9,activation='softmax', name = 'age')(dense_6)

model = Model(input_,[gender,race,age])
model.compile(loss={'gender':'binary_crossentropy', 'race':'categorical_crossentropy', 'age':'categorical_crossentropy'}, optimizer='adam', metrics= ['accuracy'])
model.summary()


model.fit(x_train,[btrain_label_gender[0:num_train],train_label_race[0:num_train],train_label_age[0:num_train]],validation_data=(x_val,[bval_label_gender[0:num_val],val_label_race[0:num_val],val_label_age[0:num_val]]),epochs = 2, callbacks = [Logger(), keras.callbacks.TensorBoard(log_dir = './logs')] , verbose = False)

model.save('trained.h5')