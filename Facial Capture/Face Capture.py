import cv2 as cv

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video capture(vc)
video_capture = cv.VideoCapture(0)

def detect_bounding_box(frame):
    """Function to detect faces and draw bounding boxes around them."""
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)
    return faces

while True:
    # Read a frame from vc
    ret, frame = video_capture.read()

    # Check if the frame was read correctly
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Detect faces in the frame
    detected_faces = detect_bounding_box(frame)

    # Display the frame with bounding boxes
    cv.imshow("My Face Detection Project", frame)

    # Check for user input to quit the loop (press 'q')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release vc and close all windows
video_capture.release()
cv.destroyAllWindows()
