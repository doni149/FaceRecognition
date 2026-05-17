# FaceRecognition
//simple FaceRecognition program by doni
import cv2
import os
import urllib.request

# 1. Download model deteksi wajah bawaan OpenCV jika belum ada
xml_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_file):
    print("Mengunduh model deteksi wajah ringan...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_file)

# Muat detektor wajah
deteksi_wajah = cv2.CascadeClassifier(xml_file)

# 2. Inisialisasi Kamera
video_capture = cv2.VideoCapture(0)
print("Sistem siap. Membuka kamera... (Tekan 'q' pada jendela gambar untuk keluar)")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal mengambil gambar dari kamera.")
        break

    # Ubah ke grayscale (hitam putih) untuk pengolahan deteksi yang cepat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Cari wajah di dalam frame kamera
    wajah_terdeteksi = deteksi_wajah.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in wajah_terdeteksi:
        # Karena ini versi sederhana, setiap wajah yang terdeteksi di kamera Anda diberi nama "sesuai yang di input"
        nama = "input nama sesuai yang akan di daftarkan"

        # Gambar kotak hijau di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Gambar latar belakang teks nama
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nama, (x + 6, y + h - 6), font, 0.8, (255, 255, 255), 1)

    # Tampilkan hasil ke layar
    cv2.imshow('Face Detection Sederhana', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("Program selesai.")
