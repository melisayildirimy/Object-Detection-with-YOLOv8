import cv2
import numpy as np
from yolov8 import YOLOv8

def load_image(image_path):
    """Bir dosya yolundan bir görüntü yükler."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü '{image_path}' bulunamadı.")
    return image

def save_image(image, output_path):
    """Bir görüntüyü bir dosya yoluna kaydeder."""
    cv2.imwrite(output_path, image)

def display_image(window_name, image):
    """Bir görüntüyü bir pencerede gösterir."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_objects_in_image(image, model_path, conf_thres=0.5, iou_thres=0.5):
    """
    YOLOv8 modeli kullanarak bir görüntüde nesneleri tespit eder.
    
    Argümanlar:
        image (np.ndarray): Girdi görüntüsü.
        model_path (str): YOLOv8 modelinin yolu.
        conf_thres (float): Güven eşiği.
        iou_thres (float): Nesneler arası örtüşme (IoU) eşiği.
    
    Dönüş:
        np.ndarray: Üzerinde tespit edilen nesneler çizilmiş görüntü.
    """
    # YOLOv8 nesne dedektörünü başlat
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    
    # Nesne tespiti yap
    boxes, scores, class_ids = yolov8_detector(image)
    
    # Tespitleri görüntü üzerinde çiz
    combined_img = yolov8_detector.draw_detections(image)
    
    return combined_img

def main(image_path, model_path, output_path=None, conf_thres=0.5, iou_thres=0.5):
    """
    Görüntü tespit sürecini yöneten ana fonksiyon.
    
    Argümanlar:
        image_path (str): Girdi görüntü dosyasının yolu.
        model_path (str): YOLOv8 model dosyasının yolu.
        output_path (str, opsiyonel): Çıktı görüntü dosyasının yolu.
        conf_thres (float, opsiyonel): Güven eşiği. Varsayılan 0.5.
        iou_thres (float, opsiyonel): IoU eşiği. Varsayılan 0.5.
    """
    # Görüntüyü yükle
    image = load_image(image_path)
    
    # Görüntüde nesne tespiti yap
    combined_img = detect_objects_in_image(image, model_path, conf_thres, iou_thres)
    
    # Tespitleri içeren görüntüyü göster
    display_image("Tespit Edilen Nesneler", combined_img)
    
    # Eğer bir çıktı yolu belirtilmişse, görüntüyü kaydet
    if output_path:
        save_image(combined_img, output_path)

if __name__ == "__main__":
    # Parametreler doğrudan kod içinde veriliyor
    model_path = "models/yolov8n.onnx"
    image_path = "images/image1.jpg"
    output_path = "output/output_image1.jpg"
    conf_thres = 0.5
    iou_thres = 0.5

    main(image_path, model_path, output_path, conf_thres, iou_thres)
