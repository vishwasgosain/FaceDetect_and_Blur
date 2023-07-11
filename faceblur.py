import cv2

# Specify the trained cascade classifier
face_cascade_name = "./haarcascade_frontalface_default.xml"

# Create a cascade classifier
face_cascade = cv2.CascadeClassifier()

# Load the specified classifier
face_cascade.load(face_cascade_name)

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess the frame
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.equalizeHist(grayimg)

    # Run the face detection
    faces = face_cascade.detectMultiScale(grayimg, 1.1, 2, 0|cv2.CASCADE_SCALE_IMAGE, (30, 30))

    print("Faces detected")

    if len(faces) != 0:
        for (x, y, w, h) in faces:
            # Draw a rectangle around each detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 5)

            # Extract the face region
            sub_face = frame[y:y+h, x:x+w]

            # Apply Gaussian blur to the face region
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)

            # Merge the blurred face region back to the frame
            frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

    # Display the frame with detected faces
    cv2.imshow("Real-Time Face Detection", frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
