import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import Visualisation_Utilities

#IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

model_path = 'C:/Users/Administrator/PycharmProjects/practice_Hand_Gestures/gesture_recognizer.task'
#print("Print the model path or press Enter to set the default path")
#model_path_inp = "\n"
#input(model_path_inp)
#if model_path_inp != "\n":
    #model_path = model_path_inp
#base_options = BaseOptions(model_asset_path=model_path)

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
    mp_image = mp.Image.create_from_file('C:/Users/Administrator/PycharmProjects/practice_Hand_Gestures/Onega.jpg')
    #print("Print the image path or press Enter to set the default path")
    #mp_image_inp = "\n"
    #input(mp_image_inp)
    #if mp_image_inp != "\n":
        #mp_image = mp_image_inp

    #image = mp.Image.create_from_file(image_file_name)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mp_image)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(mp_image)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

    Visualisation_Utilities.display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
  # The detector is initialized. Use it here.
  # ...

  # Load the input image from an image file.
  #mp_image = mp.Image.create_from_file('C:/Users/Administrator/PycharmProjects/practice_Hand_Gestures/images.jpg')
  #print("Print the image path or press Enter to set the default path")
  #mp_image_inp = "\n"
  #input(mp_image_inp)
  #if mp_image_inp != "\n":
      #mp_image = mp_image_inp

# Perform gesture recognition on the provided single image.
# The gesture recognizer must be created with the image mode.
#gesture_recognition_result = recognizer.recognize(mp_image)

