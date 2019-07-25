import numpy as np
import tensorflow as tf 
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import random
import CONFIG
import csv
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


#a custom metric (sort of ) -> it is actually the accu of one hot with single label
def arg(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true, axis=-1),
                                K.argmax(y_pred, axis=-1)),
                        K.floatx())


#extact match accu   
def multi_label_accu(y_true, y_pred):
        comp = K.equal(y_true, K.round(y_pred))
        return K.cast(K.all(comp, axis=-1), K.floatx())
  
#the method to get guiding files
def gen(num, form):
        with open('t.txt', 'w') as writeFile:
                for i in range(num):
                        s = '{}.{}\t1\n'.format(i, form)
                        writeFile.write(s)

        writeFile.close()
        destination = open('test.csv', 'a')
        in_txt = csv.reader(open('t.txt', "r"), delimiter = '\t')
        out_csv = csv.writer(destination)
        first_row = ['filename', 'labels']
        out_csv.writerow(first_row)
        out_csv.writerows(in_txt)
        destination.close()
        df = pd.read_csv('test.csv')
        return df
  

def predict():
        #set up data generator
        test_datagen = ImageDataGenerator(rescale=1./255.)

        #read file
        test_generator=test_datagen.flow_from_dataframe(
                dataframe= gen(CONFIG.TEST_NUM, CONFIG.TEST_FORMAT),
                directory= CONFIG.TEST_DIR,
                x_col= "filename",
                batch_size= 1,
                seed= 42,
                shuffle= False,
                class_mode= None,
                target_size= (CONFIG.IMAGE_SIZE,CONFIG.IMAGE_SIZE))

        #load model
        model = load_model(CONFIG.MODEL, custom_objects = {'arg': arg, 'arg2': multi_label_accu})

        #to restore order
        test_generator.reset()

        #feed in data
        pred=model.predict_generator(test_generator,
        steps=test_generator.n//test_generator.batch_size,
        verbose=1)

        #numpy it
        pred = np.array(pred)

        #select the one with max prob
        am = np.argmax(pred, axis = -1)


        predictions=[]

        #get the dictionary
        labels = CONFIG.LABELS


        for i in am:
                predictions.append(labels[i])

        
        
        filenames=test_generator.filenames
        results=pd.DataFrame({"Filename":filenames,
                                "Predictions":predictions})
        results.to_csv("Predicted_labels.csv",index=False)
        
        file1 = open("../Output/Predicted_labels.txt","a")

        with open('Predicted_labels.csv', 'r') as f:
                reader = csv.reader(f)
                i = 0
                for row in reader:
                        if(i == 0):
                                i += 1
                                continue
                        s = '{}\t{}\n'.format(row[0], row[1])
                        file1.write(s)
                        
        file1.close()




if __name__ == '__main__':

        predict()
        
