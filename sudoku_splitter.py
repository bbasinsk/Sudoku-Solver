import cv2


def _answers(index):
    path = "puzzles/" + str(index) + ".sud"
    file = open(path, 'r')
    digits = []
    file = file.read()
    file = file.replace("\n", "")
    file = file.replace(" ", "")

    for index in range(0, 81):
        digit = file[index]
        digits.append(int(digit))
    return digits


for puz_ind in range(1, 31):
    image = cv2.imread('puzzles_crop/' + str(puz_ind) + '.jpg', cv2.IMREAD_GRAYSCALE)
    digits = _answers(puz_ind)

    crop_size = int((image.shape[0])/9)
    crops = []
    for col in range(9):
        for row in range(9):
            crop = image[int(col * crop_size): int(col * crop_size + crop_size),
                         int(row * crop_size): int(row * crop_size + crop_size)]
            crops.append(crop)
    for d in range(0, 81):
        dig_folder = str(digits[d])
        puz_path = '/puz' + str(puz_ind)
        cv2.imwrite('digits/' + dig_folder + puz_path + 'pos' + str(d + 1) + '.jpg', crops[d])
        # print('saved puz: ' + str(puz_ind) + ' pos: ' + str(d))

