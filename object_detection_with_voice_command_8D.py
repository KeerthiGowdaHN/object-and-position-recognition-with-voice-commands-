import cv2
import torch
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from yolov5 import YOLOv5

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load YOLOv5 model
model = YOLOv5('yolov5s.pt')  # Use a pre-trained YOLOv5 model

# Define a function to play 8D audio
def play_8d_audio(text, position):
    # Create an audio segment
    audio = AudioSegment.from_file("beep.wav", format="wav")  # Replace with your audio file
    
    if position == "left":
        audio = audio.pan(-1.0)  # Pan to the left
    elif position == "center":
        audio = audio.pan(0.0)   # Center
    else:
        audio = audio.pan(1.0)   # Pan to the right
    
    play(audio)

def get_audio_position(bbox, frame_width):
    x_center = (bbox[0] + bbox[2]) / 2
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    frame_height, frame_width = frame.shape[:2]

    for *bbox, conf, cls in results.xyxy[0].cpu().numpy():
        if conf > 0.5:  # Confidence threshold
            bbox = list(map(int, bbox))
            label = results.names[int(cls)]
            position = get_audio_position(bbox, frame_width)
            play_8d_audio(f"{label} detected", position)
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
