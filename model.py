import csv
import cv2
import argparse
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation, Dropout

filename = "./driving_log.csv"
data = []
## Code to read data from log csv - contains anti-clockwise driving data
with open(filename,"r") as f:
    training_data = csv.reader(f)
    for row in training_data:
        data.append(row)

## Code to read data from log1 csv - contains clockwise driving data
with open("./driving_log1.csv","r") as f:
    training_data = csv.reader(f)
    for row in training_data:
        data.append(row)

#print(len(data))        
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
        
        #append angle based on camera postion
        if i == 1:
            steering_angles.append(steering_angle + 0.2)
        elif i == 2:
            steering_angles.append(steering_angle - 0.2)
        else:
            steering_angles.append(steering_angle)

        if i == 0:
            au_image = cv2.flip(resized, 1)
            images.append(au_image)
            steering_angles.append(-steering_angle)
       # elif i == 1:
       #     au_image = cv2.flip(resized, 1)
       #     images.append(au_image)
       #     steering_angles.append(-(steering_angle + 0.2))
       # elif i == 2:
       #     au_image = cv2.flip(resized, 1)
       #     images.append(au_image)
       #     steering_angles.append(-(steering_angle - 0.2))
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
            yield shuffle(X_train, y_train)
                
train_generator = generator(train_data,128)           
validation_generator = generator(validation_data,128)

def model(loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model


model = model()
model.summary()

model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 epochs=10,
                                 steps_per_epoch=len(train_data) * 10 // 128,
validation_steps=len(validation_data) * 10 // 128,verbose=1)
model.save('model.h5')

        
