
"""
Created on Wed Nov 17 02:18:45 2021

@author: Maher
"""
#  Prepear General Library 
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization


##-------------------------------------------------------------------
# import keras library 
import keras

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
# from keras.layers.normalization import BatchNormalization
import os
# print(os.listdir("../input/ck/"))
##---------------------------------------------------------------

# Load and Prepare Dataset
data_path = 'CK+_Dataset'
data_dir_list = os.listdir(data_path)

# img_rows=256
# img_cols=256
# num_channel=1

num_epoch=10

img_data_list=[]
data_y = []

i=0
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        data_y.append(i)
    i=i+1
data_x = np.array(img_data_list)
data_y = np.array(data_y)

data_x = data_x.astype('float32')
data_x = data_x/255
data_x.shape
data_y.shape

#%%%
num_classes = 7
names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

def getLabel(id):
    return ['anger','contempt','disgust','fear','happy','sadness','surprise'][id]


# Y = np_utils.to_categorical(labels, num_classes)
Y = np_utils.to_categorical(data_y, num_classes)

#Shuffle the dataset
x,y = shuffle(data_x,Y, random_state=2)
# Split the dataset
# X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=2)

# Maher add 
X_train2, x_test, y_train2, y_test = train_test_split(x, y, test_size=0.75, random_state=2)
X_train, x_val, y_train, y_val = train_test_split(X_train2, y_train2, test_size=0.25, random_state=2)

number_train_data=np.size(y_train2,0);
number_test_data=np.size(y_test,0);


##-------------------------------------------------------------------------
#build archetecture of cnn

input_shape=(48,48,3)

model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')


model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

from keras import callbacks
filename='model_train_new.csv'
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint]
callbacks_list = [csv_log]

hist = model.fit(X_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
#__________ Maher ADD ________________
from sklearn.metrics import confusion_matrix ,classification_report

Y_pred=model.predict(x_test)
y_pred = np.argmax(Y_pred, axis = 1)

Y_test = np.argmax(y_test, axis = 1)
cm = confusion_matrix(Y_test, y_pred)

print(cm)
classfi_report=classification_report(Y_test, y_pred);
print(classfi_report)

no_test_data=np.sum(cm,axis=0);

rows = len(cm);
colums = len(cm[0])
cmat_diag=cm*np.eye(rows,colums)
cmat_diag=  np.sum(cmat_diag,axis=0);
ratio_case=[]

for i in range(colums):
    ratio=round(cmat_diag[i]/no_test_data[i]*100,2)
    ratio_case.append(ratio.tolist())

Total_Ratio=sum(ratio_case)/7
Total_Ratio=round(Total_Ratio)
print ('Total_Ratio=',Total_Ratio)
print('ratio for each facial=','anger','contempt','disgust','fear','happy','sadness','surprise')
      
print ('ratio for each facial=',ratio_case)

  # zzzknn=[no_test_data,cmat_diag,ratio_case];
  
# -----save model ( cnn ) to disk to use another time for evaluate
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




plt.figure(1)

#%% summarize history for accuracy

plt.subplot(211)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()
