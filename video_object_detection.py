import cv2  # OpenCV kütüphanesini içe aktar

# YouTube'dan video yakalamak için özel bir fonksiyon olan cap_from_youtube modülünü içe aktar
from cap_from_youtube import cap_from_youtube  

# YOLOv8 modelini içe aktar
from yolov8 import YOLOv8  

# YouTube videosunun URL'si
videoUrl = 'https://youtu.be/fup-jRZKd1M?si=6dFiLGLLF2RHo7kO'

# YouTube videosunu yakala ve 720p çözünürlükte başlat
cap = cap_from_youtube(videoUrl, resolution='720p')

# Videoyu belirli bir süre sonra başlatmak için başlangıç zamanını belirle
start_time = 5 

# Başlangıç karesini belirtilen saniye kadar ileri al
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# YOLOv8 modelinin kaydedildiği yol
model_path = "models/yolov8n.onnx"

# YOLOv8 nesne dedektörünü başlat
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Algılanan nesneleri göstereceğimiz pencereyi oluştur
cv2.namedWindow("Nesneler", cv2.WINDOW_NORMAL)

# Video açık olduğu sürece işlem yap
while cap.isOpened():

    # Klavyeden 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Video karesini oku
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # YOLOv8 ile nesne tespiti yap
    boxes, scores, class_ids = yolov8_detector(frame)

    # Algılanan nesneleri çiz ve birleştirilmiş görüntüyü elde et
    combined_img = yolov8_detector.draw_detections(frame)
    
    # Algılanan nesneleri göster
    cv2.imshow("Nesneler", combined_img)
