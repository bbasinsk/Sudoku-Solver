import cv2


def get_one_puzzle(img):

    # Reading the image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024))
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # Get contours
    im2, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)



    cv2.imshow('threshold', image)
    cv2.waitKey(0)

get_one_puzzle('sudoku.jpg')