import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Load face images and encode them
savitha_image = face_recognition.load_image_file("S/project/20ECR126.jpg")
savitha_encoding = face_recognition.face_encodings(savitha_image)[0]

tharani_image = face_recognition.load_image_file("S/project/20ECR161.jpg")
tharani_encoding = face_recognition.face_encodings(tharani_image)[0]

sabari_image = face_recognition.load_image_file("S/project/20ECR120.jpg")
sabari_encoding = face_recognition.face_encodings(sabari_image)[0]

selvi_image = face_recognition.load_image_file("S/project/20ECR158.jpg")
selvi_encoding = face_recognition.face_encodings(selvi_image)[0]

known_face_encoding = [
    savitha_encoding,
    tharani_encoding,
    sabari_encoding,
    selvi_encoding
]

known_faces_names = [
    "20ECR126",
    "20ECR161",
    "20ECR120",
    "20ECR158"
]

students = known_faces_names.copy()

# Open video capture
video_capture = cv2.VideoCapture(0)

# Open CSV file for writing attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline="")
csv_writer = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]
            face_names.append(name)

            if name in students:
                students.remove(name)

                current_time = now.strftime("%H%M-%S")
                csv_writer.writerow([name, current_time])

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (10, 100)
                font_scale = 1.5
                font_color = (255, 0, 0)
                thickness = 3
                line_type = 2

                cv2.putText(frame, name + ' Present', bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
