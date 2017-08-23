import cv2


def main():
    get_one_puzzle('sudoku.jpg')


def get_one_puzzle(img):

    # Reading the image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (600, 600))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # Get contours
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    corners = _get_corners(contours)

    cv2.line(image, corners[0], corners[1], (255, 255, 255, 0), thickness=4)
    cv2.line(image, corners[1], corners[2], (255, 255, 255, 0), thickness=4)
    cv2.line(image, corners[2], corners[3], (255, 255, 255, 0), thickness=4)
    cv2.line(image, corners[3], corners[0], (255, 255, 255, 0), thickness=4)

    cv2.imshow('threshold', image)
    cv2.waitKey(0)


def _get_corners(contours):
    # look for the largest square in image
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    # calculate the center of the square
    M = cv2.moments(biggest)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # find the location of the four corners
    for a in range(0, 4):
        # calculate the difference between the center
        # of the square and the current point
        dx = biggest[a][0][0] - cx
        dy = biggest[a][0][1] - cy

        if dx < 0 and dy < 0:
            top_left = (biggest[a][0][0], biggest[a][0][1])
        elif dx > 0 > dy:
            top_right = (biggest[a][0][0], biggest[a][0][1])
        elif dx > 0 and dy > 0:
            bot_right = (biggest[a][0][0], biggest[a][0][1])
        elif dx < 0 < dy:
            bot_left = (biggest[a][0][0], biggest[a][0][1])

    # the four corners from top left going clockwise
    corners = [top_left, top_right, bot_right, bot_left]

    return corners
