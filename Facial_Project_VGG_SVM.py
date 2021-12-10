# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 02:18:45 2021

@author: Maher
"""
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16

# from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from tensorflow.keras.layers import BatchNormalization
import os

from keras.preprocessing import image
# from keras.applications.vgg16 import vgg16
# from keras.applications.vgg16 import preprocess_input

# print(os.listdir("../input/ck/"))
##---------------------------------------------------------------
model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
model.summary()
model_vgg16=Model(inputs=model.inputs,outputs=model.get_layer('block5_pool').output)

# Load and Prepare Dataset
data_path = 'CK+_maher2'
data_dir_list = os.listdir(data_path)
vgg16_feature_list=[]
data_y = []
i=0
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224,224))
        img_data=image.img_to_array(input_img_resize)
        img_data=np.expand_dims(img_data, axis=0)
        img_data=preprocess_input(img_data)
        vgg16_feature=model_vgg16.predict(img_data)
        vgg16_feature_np=np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        data_y.append(i)
    i=i+1   
# %%
data_x = np.array(vgg16_feature_list)
data_y = np.array(data_y)

np.savetxt('data_x.csv', (data_x), delimiter=',')
np.savetxt('data_y.csv', (data_y), delimiter=',')

out  = data_y.reshape(1, -1).T
zz=np.concatenate([data_x,out],axis=1)

# %%
# data_x=pd.read_csv('data_x.csv')
# data_x.head()

# num_classes = 7
names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

def getLabel(id):
    return ['anger','contempt','disgust','fear','happy','sadness','surprise'][id]
# Y = np_utils.to_categorical(data_y, num_classes)

#Shuffle the dataset
x,y = shuffle(data_x,data_y, random_state=2)
# Split the dataset
# Maher add 
X_train2, x_test, y_train2, y_test = train_test_split(x, y, test_size=0.40, random_state=2)
X_train, x_val, y_train, y_val = train_test_split(X_train2, y_train2, test_size=0.25, random_state=2)

# %%
# print("------------------ SVM Classfier -----------------")

# #Import svm model
# from sklearn import svm
# from sklearn.metrics import classification_report, confusion_matrix

# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# clf.fit(X_train, y_train)
# #Predict the response for test dataset
# y_pred = clf.predict(x_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

#%%
# Machine Learning Classfier 
# initializing all the model objects with default parameters

# def one_hot_encode_object_array(arr):
#     unique, ids=np.unique(arr,return_inverse=True)
#     return np_utils.to_categorical(ids,len(unique))

# y_test=one_hot_encode_object_array(y_test)
# y_test = y_test.astype(int)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

model_1 = SVC(kernel='poly', degree=8)
model_2 = xgb.XGBRFClassifier(use_label_encoder=False)
model_3 = RandomForestClassifier()

# training all the model on the training dataset
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)

# # predicting the output on the validation dataset
pred_1 = model_1.predict(x_test)
pred_2 = model_2.predict(x_test)
pred_3 = model_3.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print("------------------  SVM Classfier----------------")
# print(confusion_matrix(y_test, pred_1))
print(classification_report(y_test, pred_1))

print("------------------  xgboost Classfier----------------")
# print(confusion_matrix(y_test, pred_2))
print(classification_report(y_test, pred_2))

print("------------------  RandomForest Classfier----------------")
# print(confusion_matrix(y_test, pred_3))
print(classification_report(y_test, pred_3))

#------------------ Ensemble Classfier ------------------------
# Making the final model using voting classifier
final_model = VotingClassifier(
    estimators=[('lr', model_1), ('xgb', model_2), ('rf', model_3)], voting='hard')
 # training all the model on the train dataset
final_model.fit(X_train, y_train)
 # predicting the output on the test dataset
pred_final = final_model.predict(x_test)

print("------------------  Ensemble Classfier----------------")
print(confusion_matrix(y_test, pred_final))
print(classification_report(y_test, pred_final))
