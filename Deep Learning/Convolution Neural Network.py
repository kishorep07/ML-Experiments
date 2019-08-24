"""
Convolutional Neural Networks
Works with the digital rep of image (RGB / BW)
Steps: Convolution, Max Pooling, Flattening, Full Connection

Convolution:
Feature Detector (Kernel/Filter): usually 3x3 matrix. i/p (x) filter = feat map
reduces size dep on stride
we create multiple feat map for diff feats and apply rectifier (i/p -> conv layer -> rectifier)
feats are decided by computers, will seem irrelevant to humans
we add rectifier to inc non linearality, because imgs are non linear in nature

Max Pooling (Downsampling): (others - sum, mean etc)
the img can be distorted (stretched, squeezed, tilted etc)
feat map 2x2 poll with stride 2, take max value to get pooled feat map
this removes info to prevent overfitting

Flattening:
conv pooled to 1 long col

Full connection:
Flatten -> Fully Connected -> Output Layer (one neuron per category)
Basically adding ANN layer to CNN to detect features

Softmax: Makes the final values between 0 and 1 (Squashes)
Cross-Entropy: It is a measure of error, better for classification
"""


# Import
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #used to add fully connect network


# CNN Build - Numbers from exp, powers of 2 common practice
classifier = Sequential()   #init
classifier.add(Convolution2D(32, 3, input_shape=(64, 64, 3), activation = 'relu'))  #convolution
classifier.add(MaxPooling2D(pool_size = (2,2))) #pooling
classifier.add(Flatten())   #flattening
classifier.add(Dense(units = 128, activation = 'relu')) #full connection
classifier.add(Dense(units = 1, activation = 'relu')) #op, sig - binary o/p

# CNN Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = test_set,
        validation_steps = 800)