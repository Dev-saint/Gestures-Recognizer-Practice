import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
import Visualisation_Utilities
from PIL import Image
from tkinter import filedialog

model_path = 'C:/Users/Administrator/PycharmProjects/practice_Hand_Gestures/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:
    images = []
    results = []
    # STEP 3: Load the input image.
    path = input("Введите путь до изображения: ")
    str_tmp = path.split('/')
    str_tmp.pop()
    folder_path = '/'.join(str_tmp)
    mp_image = mp.Image.create_from_file(path)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mp_image)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(mp_image)
    hand_landmarks = recognition_result.hand_landmarks
    results.append(hand_landmarks)

    image_num = Visualisation_Utilities.display_hand_landmarks(images, results, False, folder_path, 1)

    img = mp_image.numpy_view()
    img = np.full_like(img, 255, dtype=np.uint8)
    img_PIL = Image.fromarray(img)
    filename = folder_path + "/white.png"
    img_PIL.save(filename)
    images[0] = mp.Image.create_from_file(filename)

    Visualisation_Utilities.display_hand_landmarks(images, results, True, folder_path, image_num)
