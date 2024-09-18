import cv2
import threading
import tempfile
import os
import pyttsx3
import time

# Initialize TTS engine
engine = pyttsx3.init()

# Define a custom temp directory
custom_temp_dir = tempfile.gettempdir()
if not os.path.exists(custom_temp_dir):
    os.makedirs(custom_temp_dir)

thres = 0.6  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def speak_text(text):
    engine.say(text)
    engine.runAndWait()
speak_text("Welcome! We are excited to introduce our new device designed to provide enhanced path guidance and support, making navigation easier and more accessible for everyone. Your journey towards greater independence starts here!")    

while True:
    success, img = cap.read()
    if not success:
        break
    

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            speech_output = "path guidance is terminating"
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Determine the panning based on the object's position
            center_x = box[0] + box[2] // 2
            if center_x < img.shape[1] // 3:
                position = "left"
            elif center_x > 2 * img.shape[1] // 3:
                position = "right"
            else:
                position = "center"

            detected_object = classNames[classId - 1]
            
            speech_output = f"{detected_object} on the {position}"
            threading.Thread(target=speak_text, args=(speech_output,)).start()
            
            

    cv2.imshow("Output", img)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):  # Press 'q' to quit
    
        break


speak_text("Thank you for using our device! We hope it has made your navigation experience smoother and more accessible. Have a great day, and we look forward to supporting you again soon!")
cap.release()

cv2.destroyAllWindows()
