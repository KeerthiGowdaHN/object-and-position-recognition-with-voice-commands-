# -AIML-Object-pattern-detection
Requirments need to be installed in your system:
1)Python: Install Python on your Windows 11 system.Make sure to select the
option to add Python to PATH during installation.
2) OpenCV: OpenCV is the main library used for computer vision tasks. You can
install OpenCV using pip, which is Python's package manager. Open Command
Prompt or PowerShell and run the following command:
3) pip install opencv-python-headless
4) Additional Libraries: There are no additional libraries required for the provided
code snippet. However, ensure that you have the necessary files such as
'coco.names', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt', and
'frozen_inference_graph.pb' in the same directory as your Python script.

To install opencv-python package:
• pip install opencv-python


Steps to execute the project:
1. There are two way to access the project files
• Firstly choose any directory u wish to store the project files ,right
click on the screen and select open terminal. Make sure u have git
bash installed
• And type the command git
• Due to this all the project files from github will be stored in the
directory or folder u want to save.
• Then open pycharm go to “Files” then to “Open” and select the
directory where u have stored the project folder and then click on the
project folder
• Now all the files would be imported to pycharm.
• Second way u can access the files in pycharm is by downloading the
zip of the repository .go to this link and click on code and click on last
option “download zip file”
• Extract the zip file at the desired directory .
• Then open pycharm go to “Files” then to” Open” and select the
directory where u have stored the project folder and then click on the
project folder
• All the project files will be imported.
2.Install the libraries as mentioned earlier in requirements
3.execute the code by clicking on run symbol which is situated at the right top
corner
4. Runs your Python script, which captures video from the default camera, detects
objects in real-time using the specified model and configuration, and displays the
annotated video with bounding boxes and labels.
5. Once the script is running, it will continuously detect objects in the video
stream from your camera.
6. Press 'q' on your keyboard to quit the program and close the video window.
