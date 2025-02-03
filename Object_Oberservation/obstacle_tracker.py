import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import time


class ObstacleTracker:
    def __init__(self, model_path, device, obstacle_threshold=3000, obstacle_value=1, not_obstacle_value=0.5):
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_value = obstacle_value
        self.not_obstacle_value = not_obstacle_value
        self.device = device

        self.transform = Compose([
            ToTensor()
        ])

        self.model = self.load_model(model_path, device)

    def load_model(self, model_path, device):
        """Load the MiDaS model from the given path."""
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    def resize_image(self, image, target_divisor=32):
        """Resize an image to dimensions operatable by the modle."""
        # Resize image to dimensions divisible by 32
        width, height = image.size
        new_width = (width // target_divisor) * target_divisor
        new_height = (height // target_divisor) * target_divisor
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def estimate_depth(self, image):
        """Estimate the depth map of the input image."""
        # Load the input image
        image = self.resize_image(image)
        input_image = self.transform(image).unsqueeze(0)

        # Perform depth estimation
        with torch.no_grad():
            input_image = input_image.to(self.device)
            depth = self.model(input_image)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        # Convert depth map to numpy array
        depth_map = depth.cpu().numpy()

        return self.formate_depth_estimate(depth_map)
    
    def formate_depth_estimate(self, depth_map):
        #print(depth_map)
        hight, width = depth_map.shape
        depth_map = depth_map[hight//3:2*(hight//3), :] #trim the first and last third of the image by height
        distance_points = [depth_map[:, i*(width//12):(i+1)*(width//12)] for i in range(0,12)]  #devide the image into 12 parts by width
        distance_points = [item.max() for item in distance_points]  #find the minimum depth/maximum value in each part
        distance_points = [self.obstacle_value if i < self.obstacle_threshold else self.not_obstacle_value for i in distance_points]  #assign the values to be interpreted as distances

        return distance_points




def main():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObstacleTracker(model_path, device)

    start_time = time.time()
    for img_path in images:
        # Perform depth estimation
        image = Image.open(img_path).convert("RGB")
        depth_map = model.estimate_depth(image)
        print(depth_map)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
