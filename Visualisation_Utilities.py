import math
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import brush
from tkinter import filedialog

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
sigma = 0.44
accurately_contours = []
rect = []
image_wh = []


def find_accurate_contour(img):
    global accurately_contours
    # define the upper and lower boundaries of the HSV pixel intensities
    # to be considered 'skin'
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array((0, 50, 128), dtype="uint8")
    upper = np.array((179, 255, 255), dtype="uint8")

    # накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsvim, lower, upper)
    # cv2.imshow("Thresh", thresh)

    # draw the contours on the empty image
    accurately_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    accurately_contours = max(accurately_contours, key=lambda x: cv2.contourArea(x))

    return thresh


def accurately_segmentation(img):
    global accurately_contours

    thresh = find_accurate_contour(img)
    thresh[:] = 0
    mask = cv2.drawContours(thresh, [accurately_contours], -1, 255, -1)
    img[mask != 255] = (255, 255, 255)
    return img


# Function to extract hand contour
def extract_hand_contour(hand_landmarks, width, height):
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append([x, y])
    contour = cv2.convexHull(np.array(points, dtype=np.int32))
    return contour


def hand_segmentation(image, multi_hand_landmarks_list, i):
    global rect
    cropped_image = np.ndarray
    img = image
    # Check if hands are detected
    if multi_hand_landmarks_list:
        # Convert the image to BGR color
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Iterate over detected hands
        for hand_landmarks in multi_hand_landmarks_list[i]:
            # Extract hand contour
            contour = extract_hand_contour(hand_landmarks, image.shape[1], image.shape[0])

            # Create a mask for the hand contour
            mask = np.full_like(image_bgr, 0, dtype=np.uint8)

            # Crop the image using the hand contour
            rect = cv2.boundingRect(contour)
            mask = cv2.rectangle(mask, rect, (255, 255, 255), cv2.FILLED)
            image_bgr = cv2.bitwise_and(image_bgr, mask)
            cropped_image = accurately_segmentation(image_bgr)
            x, y, w, h = rect
            img = cropped_image[y:y + h, x:x + w]

        return img, cropped_image

    else:
        return img, cropped_image


def resize_hand_image(img):
    desired_width = 200  # желаемая ширина

    # соотношение сторон: ширина, делённая на ширину оригинала
    aspect_ratio = desired_width / image_wh.shape[1]

    # желаемая высота: высота, умноженная на соотношение сторон
    desired_height = int(image_wh.shape[0] * aspect_ratio)

    dim = (desired_width, desired_height)  # итоговые размеры

    # Масштабируем картинку
    resized_img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)

    return resized_img


def save_and_display_image(image, skeleton_flag, multi_hand_landmarks_list, i):
    global image_wh
    if skeleton_flag:
        x, y, w, h = rect
        img = image[y:y + h, x:x + w]
        img = Image.fromarray(img)
        img.show()
        path = filedialog.asksaveasfilename()
        img.save(path)
        image = image_wh[y:y + h, x:x + w]
        image = brush.hand_painting(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        path = filedialog.asksaveasfilename()
        img.save(path)
        img.show()
        img = resize_hand_image(image)
        path = filedialog.asksaveasfilename()
        img = Image.fromarray(img)
        img.save(path)
        img.show()
    else:
        img = Image.fromarray(image)
        img.show()
        path = filedialog.asksaveasfilename()
        img.save(path)

        cropped_image, image_wh = hand_segmentation(image, multi_hand_landmarks_list, i)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_image)
        cropped_image.show()
        path = filedialog.asksaveasfilename()
        cropped_image.save(path)


def display_hand_landmarks(images, results, skeleton_flag):
    """Displays hand landmarks."""
    # Images and landmarks.
    images = [image.numpy_view() for image in images]
    multi_hand_landmarks_list = [multi_hand_landmarks for (multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Display hand landmarks and skeletons.
    for i, image in enumerate(images[:rows * cols]):
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        if skeleton_flag and multi_hand_landmarks_list:
            include_numeration = input("Do you want to include numeration of hand landmarks? (y/n): ")
            if include_numeration.lower() == 'y':
                for hand_landmarks in multi_hand_landmarks_list[i]:
                    for idx, landmark in enumerate(hand_landmarks):
                        # Convert normalized coordinates to pixel values
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])

                        # Add a numbered label next to each landmark
                        cv2.putText(annotated_image, str(idx), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1, cv2.LINE_AA)

        save_and_display_image(annotated_image, skeleton_flag, multi_hand_landmarks_list, i)
