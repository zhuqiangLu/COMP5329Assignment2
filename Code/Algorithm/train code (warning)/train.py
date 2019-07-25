

import numpy as np
import tensorflow as tf 
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import os
import cv2
import random
import keras.backend as K

"""# Load Data From git
* don't do it if already did
"""


"""#Define CONFIGs"""

import os

ROOT = os.getcwd()

NUM_CLASS = 20

TRAIN_LABEL_FILE = 'train.txt' 

TRAIN_LABEL_CSV = 'train.csv'

TRAIN_FILE = 'train2014'

TRAIN_DATA_DIR = os.path.join(ROOT, TRAIN_FILE)


POOLING = 'avg'

ACTIVATION = 'sigmoid'


METRICS = ['accuracy']

TRAIN_RATIO = 0.9

DEV_RATIO = 0.05

TEST_RATIO = 0.05

BATCH_SIZE = 32

EPOCH = 10

IMAGE_SIZE = 300

  
TEST_FILE = 'val2014'

"""#Data reader
* do not run this if train.csv already exists
"""

import csv

txt_file = r"train.txt"
csv_file = r"train.csv"

# use 'with' if the program isn't going to immediately terminate
# so you don't leave files open
# the 'b' is necessary on Windows
# it prevents \x1a, Ctrl-z, from ending the stream prematurely
# and also stops Python converting to / from different line terminators
# On other platforms, it has no effect
in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'a'))
first_row = ['filename', 'labels']
out_csv.writerow(first_row)
out_csv.writerows(in_txt)

"""#Generators"""

#split the labels by comma
df = pd.read_csv(TRAIN_LABEL_CSV)
df["labels"]=df["labels"].apply(lambda x:x.split(","))

#read labels
with open(TRAIN_LABEL_FILE, 'r') as f:
    labels = f.readlines()
    
#calculate the ratios
train_size = int(len(labels) * TRAIN_RATIO)
dev_size = int(len(labels) * DEV_RATIO)
test_size = int(len(labels) * TEST_RATIO)


#set up generators
datagen = ImageDataGenerator(
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     shear_range=0.2,
#     rotation_range=40,
#     horizontal_flip = True,
    rescale=1./255.)

test_datagen = ImageDataGenerator(rescale=1./255.)

#set up train generator
train_generator= datagen.flow_from_dataframe( 
    dataframe= df[:train_size],
    directory= TRAIN_DATA_DIR,
    x_col= "filename",
    y_col= "labels",
    batch_size= BATCH_SIZE,
    seed= 42,
    shuffle= True,
    class_mode= "categorical",
    target_size= (IMAGE_SIZE,IMAGE_SIZE,3))

#set up dev generator
valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df[train_size:(train_size + dev_size) ],
    directory= TRAIN_DATA_DIR,
    x_col= "filename",
    y_col= "labels",
    batch_size= BATCH_SIZE,
    seed= 42,
    shuffle= True,
    class_mode= "categorical",
    target_size= (IMAGE_SIZE,IMAGE_SIZE,3))

#set up test
test_generator=test_datagen.flow_from_dataframe(
    dataframe=df[(train_size + dev_size): ],
    directory= TRAIN_DATA_DIR,
    x_col= "filename",
    batch_size= 1,
    seed= 42,
    shuffle= False,
    class_mode= None,
    target_size= (IMAGE_SIZE,IMAGE_SIZE,3))




#get pretrain weights

"""#Body of the Transfer Learning
* uses ResNet-50
"""



base = applications.densenet.DenseNet169(include_top=False, weights='imagenet', pooling = POOLING)

#define base model
model = Sequential()


model.add(base)



model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.3))

model.add(Dense(20, activation = 'sigmoid'))


model.layers[0].trainable = False

"""#Compile model"""

def arg(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())

  
def multi_label_accu(y_true, y_pred):
    comp = K.equal(y_true, K.round(y_pred))
    return K.cast(K.all(comp, axis=-1), K.floatx())
  
  


opt = optimizers.RMSprop(lr = 0.0005)
model.compile(optimizer = opt, loss = 'binary_crossentropy' , metrics = ['accuracy', arg, multi_label_accu])

model.summary()

model.fit_generator(train_generator, 
                    steps_per_epoch = train_size//BATCH_SIZE,
                    validation_data = valid_generator,
                    validation_steps= dev_size//BATCH_SIZE,
                    epochs = 1,
                    verbose = 1
                   )

"""#Test"""

test_generator.reset()
pred=model.predict_generator(test_generator,
                             steps=test_generator.n//test_generator.batch_size,
                             verbose=1)

test_labels = df['labels'][(train_size + dev_size): ].values
df[(train_size + dev_size): ]
pred = np.array(pred)
am = np.argmax(pred, axis = -1)





predictions=[]
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
for i in am:
  predictions.append(labels[i])

#save as csv
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

#print the accuracy of one label
t = 0
for i in range(len(predictions)):
  if(predictions[i] in labels[i]):
    t += 1

print(t/len(predictions))







