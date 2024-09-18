import cv2  # Import OpenCV for computer vision tasks
import threading  # Import threading to handle TTS in parallel
import tempfile  # Import tempfile to create temporary files/directories
import os  # Import os for operating system functions
import pyttsx3  # Import pyttsx3 for text-to-speech functionality
import time  # Import time for time-related tasks

# Initialize TTS engine
engine = pyttsx3.init()

# Define a custom temp directory
custom_temp_dir = tempfile.gettempdir()  # Get the default temporary directory
if not os.path.exists(custom_temp_dir):  # Check if the temp directory exists
    os.makedirs(custom_temp_dir)  # Create the directory if it does not exist

thres = 0.6  # Set the threshold for object detection confidence

cap = cv2.VideoCapture(0)  # Initialize webcam capture
cap.set(3, 1280)  # Set the width of the frame
cap.set(4, 720)  # Set the height of the frame
cap.set(10, 70)  # Set the brightness of the frame (if supported)

# Load class names from file
classNames = []  # Initialize an empty list for class names
classFile = 'coco.names'  # Path to the file containing class names
with open(classFile, 'rt') as f:  # Open the class names file in read mode
    classNames = f.read().rstrip('\n').split('\n')  # Read and split the class names into a list

# Load the pre-trained object detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to the model configuration file
weightsPath = 'frozen_inference_graph.pb'  # Path to the model weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Load the model with the specified configuration and weights
net.setInputSize(320, 320)  # Set the input size for the model
net.setInputScale(1.0 / 127.5)  # Scale the input to the model
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean value for normalization
net.setInputSwapRB(True)  # Swap red and blue channels for the model

# Function to handle TTS
def speak_text(text):
    engine.say(text)  # Queue the text for speech
    engine.runAndWait()  # Wait until the speech is finished

# Initial welcome message
speak_text("Welcome! We are excited to introduce our new device designed to provide enhanced path guidance and support, making navigation easier and more accessible for everyone. Your journey towards greater independence starts here!")    

while True:  # Main loop to process video frames
    success, img = cap.read()  # Read a frame from the webcam
    if not success:  # Check if frame reading was unsuccessful
        break  # Exit the loop if no frame is captured
    
    # Perform object detection on the frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    # Check if any objects are detected
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            speech_output = "path guidance is terminating"  # Default speech output
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw bounding box around detected object
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display object class name
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display confidence score

            # Determine the object's position in the frame
            center_x = box[0] + box[2] // 2  # Calculate the center x-coordinate of the bounding box
            if center_x < img.shape[1] // 3:  # If object is in the left third of the frame
                position = "left"
            elif center_x > 2 * img.shape[1] // 3:  # If object is in the right third of the frame
                position = "right"
            else:  # If object is in the center third of the frame
                position = "center"

            detected_object = classNames[classId - 1]  # Get the name of the detected object
            
            # Construct the speech output
            speech_output = f"{detected_object} on the {position}"
            threading.Thread(target=speak_text, args=(speech_output,)).start()  # Start a new thread for TTS

    cv2.imshow("Output", img)  # Display the processed frame

    key = cv2.waitKey(1)  # Wait for a key event (1 ms delay)
    
    if key == ord('q'):  # Check if 'q' is pressed to quit
        break

# Final thank you message
speak_text("Thank you for using our device! We hope it has made your navigation experience smoother and more accessible. Have a great day, and we look forward to supporting you again soon!")

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
