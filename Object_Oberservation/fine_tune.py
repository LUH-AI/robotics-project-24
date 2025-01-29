import os
from ultralytics import YOLO

def main():
    # Pfade zu den Trainings- und Validierungsdaten (im YOLO-Format)
    dataset_path = "Object_Observation/Data/Plant"
    data_yaml = os.path.join(dataset_path, "data.yaml")  # data.yaml definiert die Klassen und Datenpfade

    # Wähle das vortrainierte YOLO11-Modell
    model_name = "Object_Observation/yolo11n.pt"  # Alternativen: "yolo11s.pt", "yolo11m.pt", etc.

    # Modell laden
    model = YOLO(model_name)

    # Hyperparameter für das Fine-Tuning
    epochs = 50  # Anzahl der Trainings-Epochen
    batch_size = 16  # Batch-Größe
    image_size = 640  # Eingabebildgröße (z. B. 640x640 Pixel)

    # Training
    model.train(
        data=data_yaml,  # Pfad zur data.yaml-Datei
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,  # Bildgröße
        workers=4,  # Anzahl der parallelen Worker für Datenvorbereitung
        device=0  # GPU verwenden (oder "cpu" für die CPU)
    )

    # Nach dem Training: Speichern und Evaluierung
    print("Training abgeschlossen. Modell wird gespeichert...")
    model_path = os.path.join("runs", "train", "weights", "best.pt")
    print(f"Das feinabgestimmte Modell ist unter {model_path} verfügbar.")

if __name__ == "__main__":
    main()
