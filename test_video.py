import face_recognition as face 
import numpy as np 
import cv2
import os
import dlib
import threading

# ใช้สำหรับตรวจจับบุคคลในวิดีโอ

dlib.DLIB_USE_CUDA = True

# ตรวจสอบว่า dlib ใช้ GPU หรือไม่
if dlib.DLIB_USE_CUDA:
    print("dlib is using CUDA (GPU) for acceleration.")
else:
    print("dlib is using CPU for acceleration.")

# ดึงวิดีโอตัวอย่างเข้ามา, ถ้าต้องการใช้ webcam ให้ใส่เป็น 0
video_capture = cv2.VideoCapture(r"C:\Users\asus\Desktop\Face Recognition\Elon3.mp4")
video_capture.set(cv2.CAP_PROP_FPS, 30)  # ตั้งค่า frame rate ที่ต้องการ

# ใบหน้าคนที่ต้องการรู้จำเป็น reference #คนที่1
Elon_image = face.load_image_file("photos/Elon.JPG")
Elon_face_encoding = face.face_encodings(Elon_image)[0]

# ประกาศตัวแปร
face_locations = []
face_encodings = []
face_names = []
face_percent = []
# ตัวแปรนี้ใช้สำหรับคิดเฟรมเว้นเฟรมเพื่อเพิ่ม fps 
process_this_frame = True

known_face_encodings = [Elon_face_encoding]
known_face_names = ["Elon musk"]

def process_frame(frame):
    global face_locations, face_encodings, face_names, face_percent, process_this_frame

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    
    face_names = []
    face_percent = []

    if process_this_frame:
        face_locations = face.face_locations(rgb_small_frame, model="hog")

        face_encodings = face.face_encodings(rgb_small_frame, face_locations)

      
        for face_encoding in face_encodings:
            face_distances = face.face_distance(known_face_encodings, face_encoding)
            best = np.argmin(face_distances)
            face_percent_value = 1 - face_distances[best]

          
            if face_percent_value >= 0.4:
                name = known_face_names[best]
                percent = round(face_percent_value * 100, 2)
                face_percent.append(percent)
            else:
                name = "Bodyguard"
                face_percent.append(0)
            face_names.append(name)

    # สลับค่าเป็นค่าตรงข้ามเพื่อให้คิดเฟรมเว้นเฟรม
    process_this_frame =  process_this_frame

    # แสดงผลลัพธ์ออกมา
    for (top, right, bottom, left), name, percent in zip(face_locations, face_names, face_percent):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        if name == "Bodyguard":
            color = [46, 2, 209]
        else:
            color = [255, 102, 51]

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left - 1, top - 30), (right + 1, top), color, cv2.FILLED)
        cv2.rectangle(frame, (left - 1, bottom), (right + 1, bottom + 30), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "MATCH: " + str(percent) + "%", (left + 6, bottom + 23), font, 0.6, (255, 255, 255), 1)

    # แสดงผลลัพธ์ออกมา
    cv2.imshow("Video", frame)

while True:
    ret, frame = video_capture.read()
    if ret:
        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# ล้างค่าต่างๆเมื่อปิดโปรแกรม
video_capture.release()
cv2.destroyAllWindows()
