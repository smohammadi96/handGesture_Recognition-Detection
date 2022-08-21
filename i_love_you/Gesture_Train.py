
import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
import os
import cv2
from sklearn.utils import shuffle


path = './gesture/'


gestures = os.listdir(path)[1:]

x_ , y_ = [], []

for i in gestures:
    images = os.listdir(path + i)
    for j in images:
        if j == ".DS_Store":
            continue
        img_path = path + i + '/' + j
        img = cv2.imread(img_path, 0)
        img = np.array(img)
        img = img.reshape( (50,50,1) )
        img = img/255.0
        x_.append(img)
        y_.append( int(i) )

x = np.array(x_)
y = np.array(y_)
y = np_utils.to_categorical(y)
num_classes = y.shape[1]

x , y = shuffle(x, y, random_state=0)

split = int( 0.6*( x.shape[0] ) )
train_features = x[ :split ]
train_labels = y[ :split ]
test_features = x[ split: ]
test_labels = y[ split: ]


model = Sequential()

model.add( Convolution2D(32, 3, 3, input_shape = (50,50,1) ) )
model.add( Activation('relu') )

model.add( Convolution2D( 64,3,3 ) )
model.add( Activation('relu') )

model.add( MaxPooling2D( pool_size=(2,2) ) )

model.add( Convolution2D( 16, 3, 3 ) )
model.add( Activation('relu') )

model.add( Flatten() )

model.add( Dropout(0.25) )
model.add( Dense(num_classes) )

model.add( Activation('softmax') )

model.summary()


model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( train_features, train_labels, validation_data=( test_features, test_labels ), shuffle=True, batch_size=128, nb_epoch=3 )

model.save('Gesture_Recognize.h5')

