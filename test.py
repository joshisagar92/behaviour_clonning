import csv
import cv2
import argparse
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation, Dropout
import matplotlib.pyplot as plt

filename = "./driving_log.csv"
data = []
with open(filename,"r") as f:
    training_data = csv.reader(f)
    int index = 0
    for row in training_data:
        for i in range(3):
            path_array = row[i].split("/")
            file_name =  "{}.jpg".format(index)
            index++
            newpath = path_array[1]+"/"path_array[2] + file_name
        data.append(row)

shuffle(data)        
train_data, validation_data = train_test_split(data, test_size=0.3)  

#print(len(train_data))

def augmentImage(batch_sample):
    steering_angle = np.float32(batch_sample[3])
    images, steering_angles = [], []
    for i in range(3):
        #print(batch_sample[i])
        image = cv2.imread(batch_sample[i])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped = rgb_image[60:130, :]
        resized = cv2.resize(cropped, (160, 70))

        images.append(resized)

        if i == 1:
            steering_angles.append(steering_angle + 0.2)
        elif i == 2:
            steering_angles.append(steering_angle - 0.2)
        else:
            steering_angles.append(steering_angle)

        if i == 0:
            au_image = cv2.flip(resized, 1)
            print("0 au_image")
            images.append(au_image)
            steering_angles.append(-steering_angle)
        elif i == 1:
            au_image = cv2.flip(resized, 1)
            images.append(au_image)
            steering_angles.append(-(steering_angle + 0.2))
        elif i == 2:
            au_image = cv2.flip(resized, 1)
            images.append(au_image)
            steering_angles.append(-(steering_angle - 0.2))
    return images, steering_angles        
    

def generator(sample,batch_size = 128):
    num_sample = len(sample)
    while True:
        shuffle(sample)
        
        for offset in range(0,num_sample,batch_size):
            batch_samples = sample[offset:offset + batch_size]
            images, steering_angles = [], []
            
            for batch_sample in batch_samples:
                #print(batch_sample)
                augmented_images, augmented_angles = augmentImage(batch_sample)
                #print(augmented_images)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)
            X_train, y_train = np.array(images), np.array(steering_angles)
            return shuffle(X_train, y_train)
                
train_generator = generator(train_data[:1],128) 
print(train_generator[0])
print(train_generator[0].shape)

for i in range(6):
    #plt.imshow(train_generator[0][i])
    #plt.show()
    #plt.plot(train_generator[0][i].reshape(70,160))
    #plt.savefig('foo {}.png',i)
    print(train_generator[1][i])
    
    
#validation_generator = generator(validation_data,128)



        
