import cv2

# Initialize the video stream
video_cap = cv2.VideoCapture(0)

# Read the first frame
ret, prev_frame = video_cap.read()
if not ret:
    print("Error: Failed to capture the initial frame")
    exit()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read a new frame from the video stream
    ret, frame = video_cap.read()
    if not ret:
        print("Error: Failed to capture a frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Apply a threshold to identify regions of significant change
    _, thresh = cv2.threshold(frame_diff, 35, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected motion
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)

    # Display the frame with motion detection
    cv2.imshow("Motion Detection", frame)

    # Update the previous frame
    prev_gray = gray

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_cap.release()
cv2.destroyAllWindows()
