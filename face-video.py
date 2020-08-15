import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_alt.xml')
# face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_alt2.xml')
# face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_alt_tree.xml')
# face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_eye.xml')
# face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_smile.xml')

# CUDA
# face_cascade = cv2.CascadeClassifier('model/haarcascades_cuda/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture('image/vtest.avi')

# RTSP source
# cap = cv2.VideoCapture('rtsp://192.168.10.20:8080/video/h264')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()