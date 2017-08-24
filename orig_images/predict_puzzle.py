import numpy as np
import cv2


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
        crop_input.reshape(crop_size, crop_size, 1)
        prediction = model.predict(crop_input)
        predictions.append(prediction)
    print(predictions)
