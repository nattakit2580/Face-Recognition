import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os  # เพิ่มโมดูล os

# #โค้ดนี้สามารถตรวจจับได้แค่ คนละ 1 รอบ

# เปิดกล้องวิดีโอ
video_capture = cv2.VideoCapture(0)

# กำหนด path สำหรับไฟล์รูปภาพ
photo_path = r"C:\Users\asus\Desktop\Face Recognition\photos"

# โหลดรูปภาพใบหน้าสำหรับการจดจำ
bill_gates_encoding = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(photo_path, "captured_1.jpg")))[0]
elon_encoding = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(photo_path, "Elon.JPG")))[0]

# กำหนดรายชื่อและข้อมูลใบหน้าที่รู้จัก
known_face_encoding = [bill_gates_encoding, elon_encoding]
known_faces_names = ["bill gates", "elon musk"]

# สำหรับการตรวจสอบใบหน้าที่ได้รับความรู้
students = known_faces_names.copy()

# สร้างตัวแปรสำหรับการตรวจจับใบหน้า
face_locations = []
face_encodings = []
face_names = []

# เตรียมเวลาปัจจุบัน
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# เปิดไฟล์ CSV เพื่อเขียนข้อมูลการบันทึก
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# วนลูปการตรวจจับใบหน้า
while True:
    # อ่านภาพจากกล้อง
    _, frame = video_capture.read()
    
    # ลดขนาดภาพเพื่อความเร็วในการประมวลผล
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # แปลง BGR เป็น RGB เนื่องจาก face_recognition ใช้ RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # ตรวจจับใบหน้า
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # ทำการเปรียบเทียบใบหน้าที่ตรวจพบกับใบหน้าที่รู้จัก
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)
        
        # ถ้าพบใบหน้าที่ตรงกัน
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

            # ตรวจสอบว่ามีการตรวจจับใบหน้าของคนนี้ในรอบก่อนหน้าหรือไม่
            if name in students:
                students.remove(name)
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                lnwriter.writerow([name, current_date, current_time])

                # วาดสี่เหลี่ยมรอบใบหน้า
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                
                # กำหนดฟอนต์และข้อความ
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} - {current_time}"
                cv2.putText(frame, text, (left * 4 + 6, bottom * 4 - 6), font, 0.5, (255, 255, 255), 1)
    
    # แสดงภาพผลลัพธ์
    cv2.imshow("attendance system", frame)
    
    # ตรวจสอบการกดปุ่ม 'q' เพื่อออกจากระบบ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
video_capture.release()

# ปิดหน้าต่างทุกอย่าง
cv2.destroyAllWindows()

# ปิดไฟล์ CSV
f.close()
