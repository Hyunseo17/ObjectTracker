import cv2

# Reads the frams from the highway.mp4 video
cap = cv2.VideoCapture("highway.mp4")

#Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=1000)

while True:
  ret, frame = cap.read()
  height, width, _ = frame.shape
  # Extract region of interest
  roi = frame[100: 700,550: 1000]

  # Object Detection
  mask = object_detector.apply(roi)
  _, mask = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    # Calculate the area and remove the small bits and area that doesnt refer to the objects
    area = cv2.contourArea(cnt)
    if area > 10:
      #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
      x,y,w,h=cv2.boundingRect(cnt)
      cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
  # Creates a new window that plays the video 
  cv2.imshow("ROI", roi)
  cv2.imshow("Frame", frame)
  cv2.imshow("Mask", mask)

  key = cv2.waitKey(30)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()