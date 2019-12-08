import csv
import cv2
import numpy as np
from glob import glob
import os
import csv



################################################################################
#           python generator logic to load images on the fly + augmentation logic
################################################################################
#from sklearn.model_selection import train_test_split
#from random import shuffle
#train_samples, validation_samples = train_test_split( lines, test_size=0.2 )



from keras.models import Sequential
from keras.layers import Flatten, Lambda,  Dense, Dropout, Activation
from keras.layers import Dropout, Conv2D, Convolution2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard

#tlclasses = [ TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN ]
categories = [ [1,0,0] , [0,1,0], [0,0,1]]

features = []
labels = []



for i in range(1):
    paths = glob(os.path.join('/home/student/code/sim_images/Red', '*.jpg'))
    for path in paths:
        img = cv2.imread(path)
        features.append(img)
        labels.append( categories[0] )
    paths = glob(os.path.join('/home/student/code/sim_images/Yellow', '*.jpg'))
    for path in paths:
        img = cv2.imread(path)
        features.append(img)
        labels.append( categories[1] )
    paths = glob(os.path.join('/home/student/code/sim_images/Green', '*.jpg'))
    for path in paths:
        img = cv2.imread(path)
        features.append(img)
        labels.append( categories[2] )


features = np.array(features)
labels = np.array( labels)
print("added {} features/labels".format(len(features)))

if True:
	################################################################################
	#       normalization
	################################################################################

	model = Sequential()
	model.add( Lambda(lambda x: x/255.0 - 0.5, input_shape=(32,32,3)))
	model.add( Conv2D( 6, (5, 5), padding='same', input_shape=(32,32,3), activation='relu') )
	model.add( MaxPooling2D() )
	model.add( Conv2D( 6, (5, 5), padding='same',  activation='relu') )
	model.add( MaxPooling2D() )
	model.add( Flatten())
	model.add( Dense(120) )
	model.add( Dense(60) )
	model.add( Activation("relu") )

	#softmax classifier
	model.add(Dense(3))
	model.add(Activation("softmax"))



if True:
	################################################################################
	#       keras run         
	################################################################################
	model.compile(loss='categorical_crossentropy', optimizer='adam' )

	print(features.shape)
	print(labels.shape)

	model.fit( features, labels, epochs=52, validation_split=0.3, shuffle=True )

	#model.fit_generator( trainData, trainLabels, train_generator, samples_per_epoch=samples_per_epoch, 
	#                    validation_data=validation_generator, nb_val_samples=nb_val_samples, 
	#                    nb_epoch=3)

	model.save('model_sim.h5')
