# importing the required packages
import pyautogui
import face_recognition
import cv2
import time
import serial
from scipy.spatial import distance as dist
from datetime import datetime
from datetime import date
import requests

import numpy as np
ser = serial.Serial()
ser.baudrate = 9600
ser.port = "COM3"
ser.open()

manan_image = face_recognition.load_image_file("manan.jpg")
mom_image = face_recognition.load_image_file("anita.jpg")
ramesh_image = face_recognition.load_image_file("ramesh.jpg")
dad_image = face_recognition.load_image_file("dad.jpg")

obama_face_encoding = face_recognition.face_encodings(manan_image)[0]
mom_face_encoding = face_recognition.face_encodings(mom_image)[0]
ramesh_face_encoding = face_recognition.face_encodings(ramesh_image)[0]
dad_face_encoding = face_recognition.face_encodings(dad_image)[0]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

data = '{"people":["fred", "mark", "andrew"]}'

response = requests.put('https://jsonblob.com/api/jsonBlob/f3a76b25-9ab7-11eb-84c0-63657ea94f77', headers=headers, data=data)
# Create an Empty window
counter = 0
def get_ear(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Resize this window

while True:

    # Take screenshot using PyAutoGUI
    img = pyautogui.screenshot()

    # Convert the screenshot to a numpy array
    frame2 = np.array(img)
    frame1 = frame2
    # Convert it from BGR(Blue, Green, Red) to
    # RGB(Red, Green, Blue)
    known_face_encodings = [
        obama_face_encoding,
        mom_face_encoding,
        ramesh_face_encoding,
        dad_face_encoding
    ]
    known_face_names = [
        "Manan Gupta",
        "Anita Gupta",
        "Manish Gupta",
        "Ramesh"
    ]

    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(frame2)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame2)
        face_encodings = face_recognition.face_encodings(frame2, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    # Display the results
    y = False
    y2 = False
    process = True
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face
        cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame2, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name != "Unknown":
            print(name)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            today = date.today()
            print("Today's date:", today)
        closed_count = 0
        if name == "Manan Gupta" or name == "Anita Gupta":
            counter = counter + 1
            y = True
            if process:
                face_landmarks_list = face_recognition.face_landmarks(frame1)

                # get eyes
                for face_landmark in face_landmarks_list:
                    left_eye = face_landmark['left_eye']
                    right_eye = face_landmark['right_eye']

                    color = (255, 0, 0)
                    thickness = 2

                    cv2.rectangle(frame2, left_eye[0], right_eye[-1], color, thickness)


                    ear_left = get_ear(left_eye)
                    ear_right = get_ear(right_eye)
                    diff = (ear_left - ear_right)
                    pos = diff
                    diff = diff*100
                    diff = (diff*diff)
                    closed = False

                    if diff > 5 and pos > 0 and (ear_left < 0.30 or ear_right < 0.30):
                        closed = True
                    if ear_left > 0.30 and ear_right > 0.30:
                        closed = False
                    print(ear_left)
                    print(ear_right)
                    print(pos)
                    print(diff)
                    if (closed):
                        closed_count += 1
                        print("closed")
                        y2 = True
                    else:
                        closed_count = 0
                        print("open")
                    break

    cv2.imshow('Live', frame2)
    if cv2.waitKey(1) == ord('q'):
        break
    if y and y2 and counter > 2:
        counter = 0
        print("Gate Opened")
        ser.write(b'1776')
        time.sleep(30)



cv2.destroyAllWindows()


EYES_CLOSED_SECONDS = 5




