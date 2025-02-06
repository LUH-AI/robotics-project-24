#type: ignore

import math
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Konfiguration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./runs/detect/train/weights/best.pt"  # Pfad zum trainierten YOLO-Modell
dataset_path = "./Data/Test/Test_plant.v1i.yolov11/test"  # Pfad zum Test-Datensatz
images_path = os.path.join(dataset_path, "images")

# Parameter
conf_threshold = 0.5
real_pot_width_cm = 30  # Breite des Topfes in cm
focal_length = 800  # Beispielwert für die Kamerafokallänge (angepasst an die Kalibrierung)
image_center_x = 640  # Beispielwert für Bildmitte in Pixeln (angepasst an die Kameradaten)
field_of_view = 120  # Sichtfeld der Kamera in Grad

map_size = 1000

def calculate_distance(pot_width_pixels):
    """Berechnet die Entfernung anhand der Breite des Topfes in Pixeln."""
    if pot_width_pixels == 0:
        return float('inf')
    return (real_pot_width_cm * focal_length) / pot_width_pixels

def calculate_angle(x_center, image_width):
    """Berechnet den Winkel eines Objekts relativ zur Bildmitte."""
    relative_x = x_center - (image_width / 2)
    angle = (relative_x / image_width) * field_of_view
    return -angle

def draw_grid(map_image, cell_size):
    """Zeichnet ein Schachbrettraster auf die Karte."""
    for x in range(0, map_size, cell_size):
        cv2.line(map_image, (x, 0), (x, map_size), (50, 50, 50), 1)  # Vertikale Linien
    for y in range(0, map_size, cell_size):
        cv2.line(map_image, (0, y), (map_size, y), (50, 50, 50), 1)  # Horizontale Linien

def update_local_map(robot_position, plants, pot_positions):
    """Aktualisiert die lokale Karte mit den Positionen des Roboters und der Pflanzen."""
    map_image = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    cell_size = int(map_size / (map_size / 100))  # 1m Abstand in Pixel (angepasst an die Skalierung)
    draw_grid(map_image, cell_size)

    robot_x, robot_y = robot_position

    # Roboter zeichnen
    cv2.circle(map_image, (robot_x, robot_y), 10, (255, 0, 0), -1)  # Roboter als blauer Kreis

    # Pflanzen zeichnen
    for plant in plants:
        distance, angle = plant
        angle += 90
        plant_x = int(robot_x + distance * math.cos(math.radians(angle)))
        plant_y = int(robot_y - distance * math.sin(math.radians(angle)))
        #plant_x = max(0,min(500,plant_x))
        #plant_y = max(0,min(500,plant_y))
        print("Plant distance: ",distance, " Angle: ", angle, "X: ",plant_x," Y: ",plant_y)
        cv2.circle(map_image, (plant_x, plant_y), 5, (0, 255, 0), -1)  # Pflanzen als grüne Kreise

    # Töpfe zeichnen
    for pot in pot_positions:
        distance, angle = pot
        angle += 90
        pot_x = int(robot_x + distance * math.cos(math.radians(angle)))
        pot_y = int(robot_y - distance * math.sin(math.radians(angle)))
        #pot_x = max(0,min(500,pot_x))
        #pot_y = max(0,min(500,pot_y))
        print("Pot distance: ",distance, " Angle: ", angle, "X: ",pot_x," Y: ",pot_y)
        cv2.circle(map_image, (pot_x, pot_y), 5, (0, 0, 255), -1)  # Topf als roter Kreis

    cv2.imshow("Lokale Karte", map_image)

def main():  # noqa: D103
    # YOLO-Modell laden
    model = YOLO(model_path)

    # Roboterposition (unten in der Mitte der Karte)
    robot_position = (int(map_size/2), int(map_size)-50)

    cv2.namedWindow("Erkannte_Objekte", cv2.WINDOW_AUTOSIZE)

    # Alle Bilder im Ordner durchlaufen
    for filename in os.listdir(images_path):
        image_path = os.path.join(images_path, filename)

        # Bild laden
        image = cv2.imread(image_path)
        print("image_path", image_path)

        # Prediction durchführen
        results = model(image)
        plants = []
        pot_positions = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()

                if confidence > conf_threshold:
                    label = f"Class {int(cls)}: {confidence:.2f}"
                    color = (0, 255, 0)  # Grün
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    x_center = (x1 + x2) / 2
                    angle = calculate_angle(x_center, image.shape[1])

                    # Entfernungsschätzung für den Topf basierend auf der Bounding Box
                    if cls == 1:  # Klasse 0 ist der Blumentopf (angepasst an die Klassendefinition)
                        pot_width_pixels = x2 - x1
                        distance = calculate_distance(pot_width_pixels)
                        pot_positions.append((distance, angle))

        # Lokale Karte aktualisieren
        update_local_map(robot_position, plants, pot_positions)

        # Bild anzeigen
        cv2.imshow("Erkannte_Objekte", image)

        # Warten, bis eine Taste gedrückt wird
        print(f"Zeige Bild: {filename}. Drücke eine Taste, um zum nächsten Bild zu wechseln.")
        cv2.waitKey(0)

    # OpenCV-Fenster schließen
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
