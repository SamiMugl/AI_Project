# face_landmark_detection
Facial Landmarks Detection using dlib

Using dlib and opencv to detect the 68 facial landmarks on each face present in the image

Prgramming Language used: Python 3.6 
Libraries Used: Numpy, Pandas, cv2, json, dlib

To install dlib follow these instructions:

1. Install cmake by typing on the command line: pip install cmake
2. Install visual studio
3. In Visual Studio go to the Individual Components tab, Visual C++ Tools for Cmake, and check the checkbox
4. Install dlib by typing on the command line: pip install dlib

Download the file named 'shape_predictor_68_face_landmarks.dat' and place in the same directory as the face_landmark_detection.py file. This contains the training data for the predictor to train and then later predict on the image we give as an input.

Code takes the path on the image containing face(s) as an input and creates Json and excel file as outputs. Output files contain landmark coordinates of all 68 landmarks on each face in the picture.

Additional Implementation: An image with the curve along the jawline of each face in the image is also opened up in a different window
