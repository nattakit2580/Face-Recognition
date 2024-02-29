import cv2
import os

# ฟังก์ชันสำหรับสร้างโฟลเดอร์สำหรับบุคคล
def create_person_folder(person_name, base_dir):
    # สร้างโฟลเดอร์สำหรับบุคคลหากยังไม่มี
    person_dir = os.path.join(base_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    return person_dir

# ตำแหน่งของโฟลเดอร์หลักที่ใช้เก็บรูปภาพ
base_directory = "C:\\Users\\asus\\Desktop\\Face Recognition\\photos"

# เปิดกล้องเว็บแคม
video_capture = cv2.VideoCapture(0)

# ให้ผู้ใช้ป้อนชื่อของบุคคลที่ต้องการบันทึกรูปภาพ
person_name = input("Enter person's name: ")

# สร้างโฟลเดอร์สำหรับบุคคล
person_folder = create_person_folder(person_name, base_directory)

# ตัวนับสำหรับการบันทึกรูป
capture_count = 0

while True:
    _, frame = video_capture.read()
    cv2.imshow("Press 'c' to capture", frame)

    # ตรวจสอบการกดปุ่มบนคีย์บอร์ด
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # บันทึกรูปที่ถูกจับไว้
        capture_count += 1
        image_path = os.path.join(person_folder, f"captured_{capture_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image {capture_count} captured and saved for {person_name}.")

        # หยุดลูปหากบันทึกรูปไปแล้ว 1 รูป
        if capture_count == 1:
            break

    elif key == ord('q'):
        break

# ปิดกล้องเว็บแคมและหน้าต่างทั้งหมด
video_capture.release()
cv2.destroyAllWindows()
