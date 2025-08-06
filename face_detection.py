import cv2

# Correct XML file name (watch the extension!)
alg = "haarcascade_frontalface_default.xml"

# Load the cascade classifier
haar_cascade = cv2.CascadeClassifier(alg)

# Open webcam
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()  # Make sure you get the return value 'ret'

    if not ret:
        print("Failed to grab frame from camera.")
        break

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = haar_cascade.detectMultiScale(grayimg, 1.3, 4)  # No space in 1.3

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Proper rectangle

    cv2.imshow("FaceDetection", img)

    key = cv2.waitKey(10)
    if key == 27:  # 27 = ESC key
        break

cam.release()
cv2.destroyAllWindows()

