from keras.datasets import cifar10
import numpy as np
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from tensorflow.keras.optimizers import SGD
from keras.layers import BatchNormalization
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# load data - 50k training samples and 10k test samples
# 32x32 pixel images - 10 output classes (labels)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# one-hot encoding for the labels (1,2 ...) will be replaced by arrays with 1s and 0s
# 0 - [1,0,0,0,0,0,0,0,0]
# 1 - [0,1,0,0,0,0,0,0,0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# normalize the data (test and training set as well)
X_train = X_train / 255.0
X_test = X_test / 255.0

# construct the CNN model
# construct the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# training the model
optimizer = SGD(learning_rate=0.001, momentum=0.95)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# evaluate model
model_result = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy of CNN model: %s' % (model_result[1] * 100.0))







