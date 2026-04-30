
import cv2

# Start video capture (0 = webcam)
cap = cv2.VideoCapture(0)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Remove noise using threshold
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small objects/noise
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Motion Tracking", frame)
    cv2.imshow("Foreground Mask", thresh)

    # Press ESC key to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()