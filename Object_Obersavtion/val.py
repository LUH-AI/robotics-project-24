import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from ultralytics import YOLO

# Konfiguration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "runs/detect/train/weights/best.pt"  # Pfad zum trainierten YOLO11-Modell
dataset_path = "Data/Test/Test_plant.v1i.yolov11/test"  # Pfad zum Test-Datensatz
confidence_threshold = 0.5
iou_threshold = 0.5

# YOLO11 Modell laden
def load_model(model_path):
    model = YOLO(model_path)  # Erstelle ein Modell-Objekt
    #checkpoint = torch.load(model_path, map_location=device)  # Lade die Gewichte
    #model.load_state_dict(checkpoint)  # Gewichte in das Modell laden
    #model = model.to(device)  # Modell auf das Gerät verschieben
    #model.eval()  # Modell in den Evaluierungsmodus versetzen
    return model

# Vorhersagen verarbeiten
def process_predictions(predictions, confidence_threshold):
    filtered_predictions = []
    for pred in predictions:
        if pred[4] > confidence_threshold:  # Confidence-Threshold anwenden
            filtered_predictions.append(pred)
    return torch.tensor(filtered_predictions)

# Ground Truth im YOLO-Format laden
def load_ground_truth(label_file):
    ground_truth_boxes = []
    with open(label_file, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # YOLO-Format in Bounding Box [x1, y1, x2, y2] umwandeln
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            ground_truth_boxes.append([x1, y1, x2, y2, class_id])
    return torch.tensor(ground_truth_boxes)

# Bounding Box- und Klassenvergleich durchführen
def validate(model, dataset_path):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")
    images = os.listdir(images_path)
    total_iou = 0
    total_predictions = 0
    total_ground_truths = 0

    for image_name in tqdm(images):
        image_path = os.path.join(images_path, image_name)
        label_file = os.path.join(labels_path, image_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_file):
            print(f"Keine Labels gefunden für {image_name}")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(device)

        # Ground Truth laden
        ground_truth_boxes = load_ground_truth(label_file)

        # Vorhersagen des Modells abrufen
        with torch.no_grad():
            predictions = model(image_tensor)[0]  # YOLO gibt [x, y, w, h, confidence, class_id] zurück

        predictions = process_predictions(predictions, confidence_threshold)

        # IOU berechnen
        if len(predictions) > 0 and len(ground_truth_boxes) > 0:
            pred_boxes = predictions[:, :4]
            gt_boxes = ground_truth_boxes[:, :4]
            ious = box_iou(pred_boxes, gt_boxes)

            total_iou += ious.sum().item()
            total_predictions += len(pred_boxes)
            total_ground_truths += len(gt_boxes)

    # Durchschnittswerte ausgeben
    avg_iou = total_iou / total_predictions if total_predictions > 0 else 0
    recall = total_iou / total_ground_truths if total_ground_truths > 0 else 0

    print(f"Durchschnittliche IOU: {avg_iou:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    model = load_model(model_path)
    validate(model, dataset_path)
