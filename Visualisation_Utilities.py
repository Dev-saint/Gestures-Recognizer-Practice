import math

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
sigma = 0.44
rect = []


def count_white_points(img, folder_path):

    file_path = folder_path + "/rows.txt"
    with open(file_path, "w") as f:
        for idx in range(img.shape[0]):
            white_count = 0
            for elem in img[idx, :]:
                if np.all(elem == 255):
                    white_count += 1
            f.write(str(white_count) + "\n")
        f.close()

    file_path = folder_path + "/cols.txt"
    with open(file_path, "w") as f:
        for idx in range(img.shape[1]):
            white_count = 0
            for elem in img[:, idx]:
                if np.all(elem == 255):
                    white_count += 1
            f.write(str(white_count) + "\n")
        f.close()


def resize_hand_image(img, des_w, des_h):
    dim = (des_w, des_h)  # итоговые размеры

    # Масштабируем картинку
    resized_img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)

    return resized_img


def rotate_hand_image(img, multi_hand_landmarks_list):
    if multi_hand_landmarks_list:
        for hand_landmarks in multi_hand_landmarks_list[0]:
            # Get the coordinates of the first point and middle finger
            x0 = hand_landmarks[0].x * img.shape[1]
            y0 = hand_landmarks[0].y * img.shape[0]
            x_middle_finger = hand_landmarks[12].x * img.shape[1]
            y_middle_finger = hand_landmarks[12].y * img.shape[0]

            # Calculate the rotation angle
            c = math.sqrt((x_middle_finger - x0) ** 2 + (y_middle_finger - y0) ** 2)
            angle = math.degrees(math.asin((x_middle_finger - x0) / c))

            # Rotate the image based on the calculated angle
            height, width, _ = img.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            # Calculate the dimensions of the rotated image
            cos_theta = np.abs(rotation_matrix[0, 0])
            sin_theta = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin_theta) + (width * cos_theta))
            new_height = int((height * cos_theta) + (width * sin_theta))

            # Adjust the rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
            rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

            straightened_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
            cv2.imshow("rotated image", straightened_image)

        return straightened_image
    else:
        return None


# Function to extract hand contour
def extract_hand_contour(hand_landmarks, width, height):
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append([x, y])
    contour = cv2.convexHull(np.array(points, dtype=np.int32))
    return contour


def crop_hand_image(img):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = np.invert(gray_img)
    # apply binary thresholding
    ret, thresh = cv2.threshold(inverted, 1, 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area (optional)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    # Select the largest contour
    largest_contour = contour[0]

    rectangle = cv2.boundingRect(largest_contour)
    return rectangle


def draw_ellipse(image, width, top, bottom, color):
    # Define bottom and top points of the finger
    x_top, y_top = top
    x_bottom, y_bottom = bottom
    x_top += 10
    y_top += 10

    # Calculate the angle between the two points
    c = math.sqrt((x_top - x_bottom) ** 2 + (y_top - y_bottom) ** 2)
    angle = math.degrees(math.asin(abs(x_top - x_bottom) / c))

    # Draw the ellipse aligned with the finger direction
    if x_bottom > x_top:
        if y_bottom < y_top:
            angle = 0 - angle
    else:
        if y_bottom > y_top:
            angle = 0 - angle

    ellipse_center = ((x_bottom + x_top) // 2, (y_bottom + y_top) // 2)
    ellipse_axes = (int(width - 10) // 2, abs(y_top - y_bottom) // 2)
    image = cv2.ellipse(image, ellipse_center, ellipse_axes, int(angle), 0, 360, color, -1)
    return image


def color_hand(img, multi_hand_landmarks_list):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    colored_image = np.full_like(img, 255)
    if multi_hand_landmarks_list:
        for hand_landmarks in multi_hand_landmarks_list[0]:
            x_thumb = []
            y_thumb = []
            x_pointer_finger = []
            y_pointer_finger = []
            x_middle_finger = []
            y_middle_finger = []
            x_ring_finger = []
            y_ring_finger = []
            x_little_finger = []
            y_little_finger = []
            for idx, landmark in enumerate(hand_landmarks):
                if idx in [3, 4]:
                    x_thumb.append(int(landmark.x * colored_image.shape[1]))
                    y_thumb.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [7, 8]:
                    x_pointer_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_pointer_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [11, 12]:
                    x_middle_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_middle_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [15, 16]:
                    x_ring_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_ring_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [19, 20]:
                    x_little_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_little_finger.append(int(landmark.y * colored_image.shape[0]))

            x0 = x_thumb[0]
            x1 = x_thumb[1]
            y0 = y_thumb[0]
            y1 = y_thumb[1]
            thumb_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            x0 = x_pointer_finger[0]
            x1 = x_pointer_finger[1]
            y0 = y_pointer_finger[0]
            y1 = y_pointer_finger[1]
            pointer_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            x0 = x_middle_finger[0]
            x1 = x_middle_finger[1]
            y0 = y_middle_finger[0]
            y1 = y_middle_finger[1]
            middle_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            x0 = x_ring_finger[0]
            x1 = x_ring_finger[1]
            y0 = y_ring_finger[0]
            y1 = y_ring_finger[1]
            ring_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            x0 = x_little_finger[0]
            x1 = x_little_finger[1]
            y0 = y_little_finger[0]
            y1 = y_little_finger[1]
            little_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            hand_contour = []
            indexes = [0, 1, 5, 9, 13, 17]
            for idx in indexes:
                x = int(hand_landmarks[idx].x * colored_image.shape[1])
                y = int(hand_landmarks[idx].y * colored_image.shape[0])
                hand_contour.append([x, y])
            hand_contour = cv2.convexHull(np.array(hand_contour, dtype=np.int32))

            colored_image = cv2.fillConvexPoly(colored_image, hand_contour, (255, 255, 0))

            x0 = int(hand_landmarks[1].x * colored_image.shape[1])
            x1 = int(x_thumb[1])
            y0 = int(y_thumb[1])
            y1 = int(hand_landmarks[1].y * colored_image.shape[0])
            colored_image = draw_ellipse(colored_image, thumb_width, (x1, y1), (x0, y0), (173, 216, 230))

            x0 = int(hand_landmarks[5].x * colored_image.shape[1])
            x1 = int(x_pointer_finger[1])
            y0 = int(y_pointer_finger[1])
            y1 = int(hand_landmarks[5].y * colored_image.shape[0])
            colored_image = draw_ellipse(colored_image, pointer_width, (x1, y1), (x0, y0), (128, 0, 128))

            x0 = int(hand_landmarks[9].x * colored_image.shape[1])
            x1 = int(x_middle_finger[1])
            y0 = int(y_middle_finger[1])
            y1 = int(hand_landmarks[9].y * colored_image.shape[0])
            colored_image = draw_ellipse(colored_image, middle_width, (x1, y1), (x0, y0), (255, 0, 0))

            x0 = int(hand_landmarks[13].x * colored_image.shape[1])
            x1 = int(x_ring_finger[1])
            y0 = int(y_ring_finger[1])
            y1 = int(hand_landmarks[13].y * colored_image.shape[0])
            colored_image = draw_ellipse(colored_image, ring_width, (x1, y1), (x0, y0), (124, 252, 0))

            x0 = int(hand_landmarks[17].x * colored_image.shape[1])
            x1 = int(x_little_finger[1])
            y0 = int(y_little_finger[1])
            y1 = int(hand_landmarks[17].y * colored_image.shape[0])
            colored_image = draw_ellipse(colored_image, little_width, (x1, y1), (x0, y0), (255, 165, 0))

        return colored_image
    else:
        return None


def save_image(image, skeleton_flag, multi_hand_landmarks_list, folder_path, image_num):
    if skeleton_flag:
        img = color_hand(image, multi_hand_landmarks_list)
        image = Image.fromarray(img)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        image.save(path)

        rectangle = crop_hand_image(img)
        img = rotate_hand_image(img, multi_hand_landmarks_list)
        x, y, w, h = rectangle
        y += 40
        h += 40
        x += 20
        w += 20
        img = img[y:y + h, x:x + w]
        image = Image.fromarray(img)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        image.save(path)

        img = resize_hand_image(img, 200, 150)
        image = Image.fromarray(img)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        image.save(path)

        count_white_points(img, folder_path)
    else:
        img = Image.fromarray(image)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        img.save(path)

    return image_num


def display_hand_landmarks(images, results, skeleton_flag, folder_path, image_num):
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
            for hand_landmarks in multi_hand_landmarks_list[i]:
                for idx, landmark in enumerate(hand_landmarks):
                    # Convert normalized coordinates to pixel values
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    # Add a numbered label next to each landmark
                    cv2.putText(annotated_image, str(idx), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1, cv2.LINE_AA)

        image_num = save_image(annotated_image, skeleton_flag, multi_hand_landmarks_list, folder_path, image_num)

    return image_num
