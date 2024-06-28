import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

# Load mô hình đã huấn luyện
model = load_model('Face_data/predict/Final_test.h5')

# Định nghĩa nhãn của các lớp
labels = ['Duong', 'Long', 'Phong', 'Vu']

# Hàm dự đoán khuôn mặt từ ảnh
def predict_face(image, model, labels):
    # Hiển thị ảnh gốc
    cv2.imshow('Original Image', image)
    
    # Kiểm tra và lật ảnh nếu bị ngượcq
    image = cv2.flip(image, 1)  # Lật ảnh theo chiều ngang
    
    # Resize ảnh về kích thước 150x150
    image = cv2.resize(image, (150, 150))
    
    # Hiển thị ảnh đã resize
    cv2.imshow('Resized Image', image)
    
    image = img_to_array(image)
    image = image.reshape(1, 150, 150, 3)
    image = image.astype('float32') / 255.0
    prediction = model.predict(image)
    confidence = np.max(prediction)
    label = labels[np.argmax(prediction)]
    
    # In kết quả dự đoán và giá trị dự đoán
    print("Prediction:", prediction)
    print("Predicted Label:", label)
    print("Confidence:", confidence)
    
    return label, confidence

# Truy cập webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển ảnh về grayscale để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # Nếu không phát hiện được khuôn mặt
        cv2.putText(frame, "No Face", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label, confidence = predict_face(face, model, labels)
            # Vẽ hình chữ nhật quanh khuôn mặt và thêm nhãn dự đoán cùng tỷ lệ chính xác
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Hiển thị ảnh
    cv2.imshow('Face Recognition', frame)
    
    # Thoát bằng cách nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
