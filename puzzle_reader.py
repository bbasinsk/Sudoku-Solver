import cv2
import numpy as np


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # cv2.imshow('warped', warped)
    # cv2.waitKey(0)
    # return the warped image
    return warped


def save_one_puzzle(img, index):

    # Reading the image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (400, 600))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # Get contours
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    corners = _get_corners(contours)
    warped = four_point_transform(image, np.array(corners))
    warped = warped[16:, :]
    warped = cv2.resize(warped, (300, 300))
    if not cv2.imwrite('puzzles_crop/' + str(index) + '.jpg', warped):
        print(index)


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

for i in range(1, 61):
    # print(i)
    save_one_puzzle('orig_images/' + str(i) + '.jpg', i)
