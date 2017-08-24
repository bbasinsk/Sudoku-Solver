from PIL import Image
import numpy as np
import os

all_inputs = np.array()
all_labels = np.array()
for digit in range(0, 10):
    path = 'digits/' + str(digit) + '/'
    for filename in os.listdir(path):
        if not filename == '.DS_Store':
            img = Image.open(path + filename, mode='r')
            img_arr = np.array(img)
            all_inputs.append(img_arr)
            all_labels.append(digit)

