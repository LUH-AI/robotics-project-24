import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from ultralytics import YOLO


print("TEST2")
print("TEST1.1")
# Konfiguration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./runs/detect/train/weights/best.pt"  # Pfad zum trainierten YOLO11-Modell
dataset_path = "./Data/Test/Test_plant.v1i.yolov11/test"  # Pfad zum Test-Datensatz
images_path = os.path.join(dataset_path, "images")

print("TEST1.2")
# YOLOv11-Modell laden
model = YOLO(model_path)

# Confidence Threshold einstellen
conf_threshold = 0.5

cv2.namedWindow("Erkannte_Objekte", cv2.WINDOW_AUTOSIZE) 
# Alle Bilder im Ordner durchlaufen
for filename in os.listdir(images_path):
    image_path = os.path.join(images_path, filename)

    # Bild laden
    image = cv2.imread(image_path)
    print("image_path",image_path)
    # Prediction durchführen
    results = model(image)

    for result in results:
        boxes = result.boxes
        print("BOXES",boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() 
            cls = int(box.cls[0].cpu().numpy())         # Klassen-ID
            confidence = box.conf[0].cpu().numpy()     # Vertrauenswert
            print(x1,x2,y1,y2,cls,confidence)

            label = f"Class {int(cls)}: {confidence:.2f}"
            color = (0, 255, 0)  # Grün
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Bild anzeigen
    cv2.imshow("Erkannte_Objekte", image)

    # Warten, bis eine Taste gedrückt wird
    print(f"Zeige Bild: {filename}. Drücke eine Taste, um zum nächsten Bild zu wechseln.")
    cv2.waitKey(0)

# OpenCV-Fenster schließen
cv2.destroyAllWindows()
