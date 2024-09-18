

---

## Code Explanation

This Python script performs real-time object detection using a pre-trained SSD MobileNet v3 model with voice guidance provided by a text-to-speech (TTS) engine.

### Key Components:
1. **Text-to-Speech (TTS) Engine**:
   - The script uses `pyttsx3` to convert detected object names and their positions (left, center, right) into speech output, allowing real-time audio feedback for users.

2. **Webcam Feed**:
   - The script captures real-time video from the webcam using OpenCV (`cv2.VideoCapture(0)`).

3. **Object Detection Model**:
   - The pre-trained SSD MobileNet v3 COCO model is loaded using OpenCV’s DNN module. It detects objects in the video feed and classifies them based on the COCO dataset.

4. **Real-time Detection Loop**:
   - In each frame of the webcam feed, objects are detected, and their bounding boxes are drawn.
   - The script also determines the horizontal position (left, center, or right) of each object in the frame.
   - The detected object’s name and position are announced using the TTS engine in real-time.

5. **Threading for TTS**:
   - The script uses threading to prevent the text-to-speech function from blocking the real-time video stream, allowing smooth execution of both tasks simultaneously.

6. **Exit Mechanism**:
   - Pressing the 'q' key stops the webcam feed and exits the program.

---

## Execution Steps

To run the script, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Install Dependencies**:

   Install the required Python packages by running:

   ```bash
   pip install opencv-python pyttsx3
   ```

3. **Download the Required Model Files**:

   Ensure that the following files are placed in the same directory as the script:
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
   - `frozen_inference_graph.pb`
   - `coco.names` (for object class labels)

4. **Run the Script**:

   Execute the script using Python:

   ```bash
   python object_detection_with_voice_command_8D.py
   ```

5. **Quit the Program**:

   Press the 'q' key to stop the video feed and exit the program.

---

