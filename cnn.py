from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

batch_size = 128
num_classes = 10
epochs = 100
img_rows = 33
img_cols = 33


def predict_puzzle(model, img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    crop_size = int((image.shape[0]) / 9)
    crops = []
    for col in range(9):
        for row in range(9):
            crop = image[int(col * crop_size): int(col * crop_size + crop_size),
                         int(row * crop_size): int(row * crop_size + crop_size)]
            crops.append(crop)

    predictions = []
    for crop in crops:
        crop_input = np.array(crop) / 255
        crop_input = crop_input.reshape((1, crop_size, crop_size, 1))
        crop_input = crop_input.astype('float32')
        prediction = model.predict(crop_input)
        predictions.append(prediction)
    puzzle = []
    for i, p in enumerate(predictions):
        prediction = list(p[0])
        accuracy = max(prediction)
        guess = prediction.index(accuracy)
        puzzle.append(guess)
        print('Position:', i + 1, 'Guess:', guess, 'Confidence:', accuracy)
    for row in range(0, 9):
        for col in range(0, 9):
            print(puzzle[row*9 + col], end='.')
        print('')

all_inputs = []
all_labels = []
for digit in range(0, 10):
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels[digit] = 1
    path = 'digits/' + str(digit) + '/'
    for filename in os.listdir(path):
        if not filename == '.DS_Store':
            img = Image.open(path + filename, mode='r')
            img_arr = np.array(img) / 255
            all_inputs.append(img_arr)
            all_labels.append(labels)
all_inputs, all_labels = np.array(all_inputs), np.array(all_labels)

x_train, x_test, y_train, y_test = train_test_split(all_inputs, all_labels, test_size=0.33, random_state=42)
print('x train:', x_train.shape)
print('x test:', x_test.shape)
print('y train:', y_train.shape)
print('y test:', y_test.shape)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x train:', x_train.shape)
print('x test:', x_test.shape)
print('y train:', y_train.shape)
print('y test:', y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])


predict_puzzle(model, 'puzzles_crop/60.jpg')

