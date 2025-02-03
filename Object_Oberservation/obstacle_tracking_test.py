import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time

OBSTACLE_THRESHOLD = 3000
OBSTACLE_VALUE = 0.5


def resize_image(image, target_divisor=32):
    # Resize image to dimensions divisible by 32
    width, height = image.size
    new_width = (width // target_divisor) * target_divisor
    new_height = (height // target_divisor) * target_divisor
    return image.resize((new_width, new_height), Image.LANCZOS)

def estimate_depth(image_path, model, device):
    # Load the input image
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image)  # Ensure dimensions are divisible by 32
    input_image = transform_image(image)

    # Perform depth estimation
    with torch.no_grad():
        input_image = input_image.to(device)
        depth = model(input_image)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    # Convert depth map to numpy array
    depth_map_normalized = depth.cpu().numpy()
    #depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return depth_map_normalized

# Define transformation pipeline for MiDaS
def transform_image(image):
    transform = Compose([
        ToTensor(),
        #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def load_model(model_path):
    # Load MiDaS model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def main():
    image_path = "Object_Oberservation/Test_Image.jpeg"  # Input image
    model_path = "Object_Oberservation/model-f6b98070.pt"  # MiDaS model file

    images = ["Object_Oberservation/Data/Test/image_0.jpg", 
              "Object_Oberservation/Data/Test/image_1.jpg", 
              "Object_Oberservation/Data/Test/image_2.jpg", 
              "Object_Oberservation/Data/Test/image_3.jpg", 
              "Object_Oberservation/Data/Test/image_4.jpg", 
              "Object_Oberservation/Data/Test/image_5.jpg", 
              "Object_Oberservation/Data/Test/image_6.jpg", 
              "Object_Oberservation/Data/Test/image_7.jpg", 
              "Object_Oberservation/Data/Test/image_8.jpg", 
              "Object_Oberservation/Data/Test/image_9.jpg"]

    # Load the model
    model, device = load_model(model_path)

    # Perform depth estimation
    depth_map = estimate_depth(image_path, model, device)
    hight, width = depth_map.shape

    depth_map = depth_map[hight//3:2*(hight//3), :] #trim the first and last third of the image by height
    distance_points = [depth_map[:, i*(width//12):(i+1)*(width//12)] for i in range(0,12)] #devide the image into 12 parts by width
    distance_points = [item.max() for item in distance_points]  #find the minimum depth/maximum value in each part
    distance_points = [1 if i < OBSTACLE_THRESHOLD else 0.5 for i in distance_points]
    print(distance_points)

    # Load the original image for display
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    for i in range(0, 12):
        depth_map[:, i*(width//12):i*(width//12)+5].fill(0)

    # Plot the original image and depth map side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap="inferno")
    plt.title("Depth Map")
    plt.axis("off")

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
