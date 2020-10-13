import cv2

trained_car_data = cv2.CascadeClassifier("cars.xml")

trained_pedestrian_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")


video = cv2.VideoCapture('videoplayback.mp4')

while True:
    frame_read, frame = video.read()

    if frame_read:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    car_coordinates = trained_car_data.detectMultiScale(grayscale_frame)
    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in car_coordinates:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Car & Pedestrian Detector Project', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


print("Code Completed")