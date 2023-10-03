import numpy as np
import cv2 as cv

drawing = False
ix, iy = 0, 0
img = []
r, g, b, size = 0, 0, 0, 0


def nothing(x):
    pass


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(img, (x, y), size, (b, g, r), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.circle(img, (x, y), size, (b, g, r), -1)


def hand_painting(hand_image):
    global drawing, ix, iy, img, r, g, b, size

    # Create a black image, a window
    settings_img = np.zeros((150, 512), np.uint8)
    cv.namedWindow('settings')
    # create trackbars for color change
    cv.createTrackbar('R', 'settings', 0, 255, nothing)
    cv.createTrackbar('G', 'settings', 0, 255, nothing)
    cv.createTrackbar('B', 'settings', 0, 255, nothing)
    cv.createTrackbar('Brush size', 'settings', 1, 20, nothing)

    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1

    img = hand_image
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        cv.imshow('settings', settings_img)

        # get current positions of four trackbars
        r = cv.getTrackbarPos('R', 'settings')
        g = cv.getTrackbarPos('G', 'settings')
        b = cv.getTrackbarPos('B', 'settings')
        size = cv.getTrackbarPos('Brush size', 'settings')

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()

    return img
