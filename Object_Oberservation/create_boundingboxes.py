from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np



def draw_boxes_on_image(image, boxes):
    # Wenn das Bild ein numpy-Array ist, in ein PIL-Bild konvertieren
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Zeichnen vorbereiten
    draw = ImageDraw.Draw(image)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()

        # Rechteck zeichnen
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Text hinzufügen
        label = f"{cls} ({conf:.2f})"
        text_position = (x1, y1 - 10)
        draw.text(text_position, label, font_size=30, fill="red", hight=10)
    
    # Bearbeitetes Bild zurückgeben
    return image




if __name__ == "__main__":

    # Modell laden
    #model = YOLO('yolo11n.pt')
    model = YOLO('Object_Observation/best.pt')

    img_path = "Object_Observation/Data/Plant/test/images"
    #images = ["./Images/" + str(i) + ".jpg" for i in range(1, 22)]
    images = os.listdir(img_path)
    images = [img_path + "/" + path for path in images]
    print(images)
    #print(images)

    for i, path in enumerate(images):
        print(i)
        # Bild analysieren
        results = model(path)  # Testbild analysieren
        image = cv2.imread(path)

        # Ergebnisse auslesen
        for result in results:  # Ergebnisse für jedes Bild
            boxes = result.boxes  # Bounding-Box-Objekte
            for box in boxes:
                # Box-Koordinaten
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax
                
                # Box-Klasse und Confidence
                cls = int(box.cls[0].cpu().numpy())         # Klassen-ID
                confidence = box.conf[0].cpu().numpy()     # Vertrauenswert

                #print(f"Class: {cls}, Confidence: {confidence:.2f}, Box: {x1, y1, x2, y2}")

                label = f"Class {int(cls)}: {confidence:.2f}"
                color = (0, 255, 0)  # Grün
                #cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                #cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        draw_boxes_on_image(image, boxes).save("Object_Observation/results/" + str(i+1) + ".jpg")  
        print("Picture: ", i+1, "    cls: ", boxes.cls.cpu())  