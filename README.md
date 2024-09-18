Real-time Object Detection with YOLOv5 and 8D Audio Feedback

This project performs real-time object detection using a pre-trained YOLOv5 model and gives 8D audio feedback based on the detected object's position within the frame. The webcam captures the video feed, and when an object is detected, a corresponding beep sound is played in the left, center, or right ear depending on the object's location in the frame.

### Features:
- Real-time object detection using the YOLOv5 model.
- 8D audio feedback (left, center, right) based on object position.
- Bounding boxes and labels are drawn around detected objects.

### Requirements:
- Python 3.x
- OpenCV
- PyTorch
- YOLOv5 model (pre-trained)
- pydub for audio playback

### Steps to Execute:

1. Clone the repository:
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   

2. Install the required libraries:
   Make sure you have the necessary dependencies installed:
   pip install opencv-python torch torchvision torchaudio pydub
   

3. Install ffmpeg(required by `pydub` for audio processing):
   - For Ubuntu:
     
     sudo apt update
     sudo apt install ffmpeg
     
   - For Windows, download and install from [FFmpeg official website](https://ffmpeg.org/download.html).

4. **Download YOLOv5 Pre-trained Model**:
   The model `yolov5s.pt` is automatically downloaded if you're using the following line:
   ```python
   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
   ```

5. **Add a beep audio file**:
   Add an audio file named `beep.wav` in the root directory. You can use any short sound for the beep.

6. **Run the script**:
   Execute the script to start object detection and 8D audio feedback:
   ```bash
   python main.py
   ```

7. **Quit the Program**:
   Press `q` on the keyboard to exit the program and close the webcam feed.

---

### Additional Notes:
- Make sure your webcam is properly connected and accessible by OpenCV.
- The script will play audio panning to the left, center, or right based on the object's position relative to the camera view.

