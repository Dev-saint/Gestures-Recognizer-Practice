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


def overlay(img):
    background = np.full_like(img, 255)
    background = resize_hand_image(background, img.shape[1] + 500, img.shape[0] + 500)
    b = Image.fromarray(background)
    a = Image.fromarray(img)
    b.paste(a, (100, 100))
    b.save('fon_pillow_paste_pos.jpg', quality=95)
    background = cv2.imread('fon_pillow_paste_pos.jpg')
    return background


def rotate_hand_image(img, multi_hand_landmarks_list):
    if multi_hand_landmarks_list:
        for hand_landmarks in multi_hand_landmarks_list[0]:
            # Get the coordinates of the first point and top finger
            max_c = math.sqrt((hand_landmarks[4].x - hand_landmarks[0].x) ** 2 + (hand_landmarks[4].y -
                                                                                  hand_landmarks[0].y) ** 2)
            x1 = hand_landmarks[4].x * img.shape[1]
            y1 = hand_landmarks[4].y * img.shape[0]
            for idx in [8, 12, 16, 20]:
                current_c = math.sqrt((hand_landmarks[idx].x - hand_landmarks[0].x) ** 2 + (hand_landmarks[idx].y -
                                                                                            hand_landmarks[0].y) ** 2)
                if max_c < current_c:
                    x1 = hand_landmarks[idx].x * img.shape[1]
                    y1 = hand_landmarks[idx].y * img.shape[0]
                    max_c = current_c

            x0 = hand_landmarks[0].x * img.shape[1]
            y0 = hand_landmarks[0].y * img.shape[0]

            height, width, _ = img.shape
            h = height
            w = width
            if y1 > y0:
                img = cv2.flip(img, 0)
                height, width, _ = img.shape

            # Calculate the rotation angle
            c = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            angle = math.degrees(math.asin((x1 - x0) / c))

            # Rotate the image based on the calculated angle
            img = overlay(img)
            center = (width // 2, height // 2)
            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            cos_theta = abs(rotation_matrix[0, 0])
            sin_theta = abs(rotation_matrix[0, 1])

            # Calculate the new dimensions of the rotated image
            new_width = int((height * sin_theta) + (width * cos_theta))
            new_height = int((height * cos_theta) + (width * sin_theta))

            # Adjust the rotation matrix translation values to avoid cropping
            rotation_matrix[0, 2] += (new_width - width) // 2
            rotation_matrix[1, 2] += (new_height - height) // 2

            # Apply the rotation to the image
            rotated_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
            rotated_image = resize_hand_image(rotated_image, width, height)
            x = 100
            y = 100
            rotated_image = rotated_image[y:y + h, x:x + w]

        return rotated_image
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
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(gray_image, 1, 254)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    list_of_pts = [pt[0] for ctr in contours for pt in ctr]

    contour = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
    contour = cv2.convexHull(contour)

    rectangle = cv2.boundingRect(contour)
    return rectangle


def draw_circle_and_rectangle(img, width, x0, y0, x1, y1, color):
    radius = int(width / 2)
    center = (x1, y1)
    img = cv2.circle(img, center, radius, color, -1)
    if abs(x1 - x0) > abs(y0 - y1):
        pts = np.array([[x0, y0 - radius], [x1, y1 - radius],
                        [x1, y1 + radius], [x0, y0 + radius]])
    else:
        pts = np.array([[x0 - radius, y0], [x1 - radius, y1],
                        [x1 + radius, y1], [x0 + radius, y0]])
    img = cv2.fillConvexPoly(img, pts, color)

    return img, x1, y1


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
                if idx in [2, 4]:
                    x_thumb.append(int(landmark.x * colored_image.shape[1]))
                    y_thumb.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [6, 8]:
                    x_pointer_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_pointer_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [10, 12]:
                    x_middle_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_middle_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [14, 16]:
                    x_ring_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_ring_finger.append(int(landmark.y * colored_image.shape[0]))
                elif idx in [18, 20]:
                    x_little_finger.append(int(landmark.x * colored_image.shape[1]))
                    y_little_finger.append(int(landmark.y * colored_image.shape[0]))

            x0 = x_thumb[0]
            x1 = x_thumb[1]
            y0 = y_thumb[0]
            y1 = y_thumb[1]
            thumb_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2

            x0 = x_pointer_finger[0]
            x1 = x_pointer_finger[1]
            y0 = y_pointer_finger[0]
            y1 = y_pointer_finger[1]
            pointer_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2

            x0 = x_middle_finger[0]
            x1 = x_middle_finger[1]
            y0 = y_middle_finger[0]
            y1 = y_middle_finger[1]
            middle_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2

            x0 = x_ring_finger[0]
            x1 = x_ring_finger[1]
            y0 = y_ring_finger[0]
            y1 = y_ring_finger[1]
            ring_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2

            x0 = x_little_finger[0]
            x1 = x_little_finger[1]
            y0 = y_little_finger[0]
            y1 = y_little_finger[1]
            little_width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2

            x0 = 0
            y0 = 0
            for idx, landmark in enumerate(hand_landmarks):
                if idx != 0:
                    x1 = int(landmark.x * colored_image.shape[1])

                    y1 = int(hand_landmarks[idx].y * colored_image.shape[0])

                    if idx in [1, 2, 3, 4]:
                        if idx == 1:
                            x0 = x1
                            y0 = y1
                        colored_image, x0, y0 = draw_circle_and_rectangle(colored_image, thumb_width,
                                                                          x0, y0, x1, y1, (173, 216, 230))
                    elif idx in [5, 6, 7, 8]:
                        if idx == 5:
                            x0 = x1
                            y0 = y1
                        colored_image, x0, y0 = draw_circle_and_rectangle(colored_image, pointer_width,
                                                                          x0, y0, x1, y1, (128, 0, 128))
                    elif idx in [9, 10, 11, 12]:
                        if idx == 9:
                            x0 = x1
                            y0 = y1
                        colored_image, x0, y0 = draw_circle_and_rectangle(colored_image, middle_width,
                                                                          x0, y0, x1, y1, (255, 0, 0))
                    elif idx in [13, 14, 15, 16]:
                        if idx == 13:
                            x0 = x1
                            y0 = y1
                        colored_image, x0, y0 = draw_circle_and_rectangle(colored_image, ring_width,
                                                                          x0, y0, x1, y1, (124, 252, 0))
                    elif idx in [17, 18, 19, 20]:
                        if idx == 17:
                            x0 = x1
                            y0 = y1
                        colored_image, x0, y0 = draw_circle_and_rectangle(colored_image, little_width,
                                                                          x0, y0, x1, y1, (255, 165, 0))

            hand_contour = []
            indexes = [0, 1, 5, 9, 13, 17]
            for idx in indexes:
                x = int(hand_landmarks[idx].x * colored_image.shape[1])
                y = int(hand_landmarks[idx].y * colored_image.shape[0])
                hand_contour.append([x, y])

            hand_contour = cv2.convexHull(np.array(hand_contour, dtype=np.int32))
            colored_image = cv2.fillConvexPoly(colored_image, hand_contour, (255, 255, 0))

        return colored_image
    else:
        return None


def to_white(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.inRange(gray_image, 1, 255)
    img[thresh == 0] = 255

    return img


def save_image(image, skeleton_flag, multi_hand_landmarks_list, folder_path, image_num):
    if skeleton_flag:
        img = Image.fromarray(image)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        img.save(path)
        img = color_hand(image, multi_hand_landmarks_list)
        image = Image.fromarray(img)
        path = folder_path + "/" + str(image_num) + ".jpg"
        image_num += 1
        image.save(path)

        rectangle = crop_hand_image(img)
        img = rotate_hand_image(img, multi_hand_landmarks_list)
        img = to_white(img)
        x, y, w, h = rectangle
        y -= 125
        y = max(y, 0)
        h += 100
        x -= 75
        x = max(x, 0)
        w += 100
        img = img[y:y + h, x:x + w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
