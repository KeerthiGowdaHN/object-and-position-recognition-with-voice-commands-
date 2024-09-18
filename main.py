import cv2  # Import OpenCV library for computer vision tasks
import threading  # Import threading library to handle text-to-speech in a separate thread
import tempfile  # Import tempfile to manage temporary directories and files
import os  # Import os to interact with the operating system
import pyttsx3  # Import pyttsx3 for text-to-speech functionality
import time  # Import time for handling delays and timing

# Initialize TTS (Text-To-Speech) engine
engine = pyttsx3.init()  # Set up the pyttsx3 text-to-speech engine

# Define a custom temporary directory
custom_temp_dir = tempfile.gettempdir()  # Get the system's default temporary directory
if not os.path.exists(custom_temp_dir):  # Check if the temporary directory exists
    os.makedirs(custom_temp_dir)  # Create the directory if it does not exist

thres = 0.6  # Set the threshold for detecting objects. Objects with confidence below this value will be ignored

# Open the webcam
cap = cv2.VideoCapture(0)  # Initialize the video capture object for the default camera
cap.set(3, 1280)  # Set the width of the video frame to 1280 pixels
cap.set(4, 720)  # Set the height of the video frame to 720 pixels
cap.set(10, 70)  # Set the camera's brightness (value may vary depending on the camera)

# Load class names for object detection
classNames = []  # Create an empty list to store class names
classFile = 'coco.names'  # Path to the file containing class names for object detection
with open(classFile, 'rt') as f:  # Open the class names file in read text mode
    classNames = f.read().rstrip('\n').split('\n')  # Read the file, remove trailing newline, and split into a list of class names

# Load the object detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to the model's configuration file
weightsPath = 'frozen_inference_graph.pb'  # Path to the model's weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Load the pre-trained model with the configuration and weights
net.setInputSize(320, 320)  # Set the input size for the model (width, height)
net.setInputScale(1.0 / 127.5)  # Scale the input image for the model
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean for normalization of input image
net.setInputSwapRB(True)  # Swap the Red and Blue channels in the image

# Function to convert text to speech
def speak_text(text):
    engine.say(text)  # Queue the text to be spoken by the TTS engine
    engine.runAndWait()  # Block and wait for the speech to be completed

# Initial greeting message
speak_text("Welcome! We are excited to introduce our new device designed to provide enhanced path guidance and support, making navigation easier and more accessible for everyone. Your journey towards greater independence starts here!")    

while True:  # Start an infinite loop to continuously capture frames from the webcam
    success, img = cap.read()  # Capture a frame from the webcam
    if not success:  # Check if the frame was successfully captured
        break  # Exit the loop if frame capture fails

    classIds, confs, bbox = net.detect(img, confThreshold=thres)  # Detect objects in the frame with a confidence threshold

    if len(classIds) != 0:  # Check if any objects were detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw a green rectangle around the detected object
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Add the class name text on the frame
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Add the confidence score text on the frame

            # Determine the object's horizontal position
            center_x = box[0] + box[2] // 2  # Calculate the center x-coordinate of the bounding box
            if center_x < img.shape[1] // 3:
                position = "left"  # Object is on the left side of the frame
            elif center_x > 2 * img.shape[1] // 3:
                position = "right"  # Object is on the right side of the frame
            else:
                position = "center"  # Object is in the center of the frame

            detected_object = classNames[classId - 1]  # Get the name of the detected object
            
            # Prepare the speech output
            speech_output = f"{detected_object} on the {position}"
            # Create a new thread to speak the text without blocking the video stream
            threading.Thread(target=speak_text, args=(speech_output,)).start()
            
    cv2.imshow("Output", img)  # Display the processed video frame

    key = cv2.waitKey(1)  # Wait for a key press for 1 millisecond
    if key == ord('q'):  # Check if the 'q' key is pressed
        break  # Exit the loop if 'q' is pressed

# Final thank you message
speak_text("Thank you for using our device! We hope it has made your navigation experience smoother and more accessible. Have a great day, and we look forward to supporting you again soon!")

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
