# Normalize and denormalize an image

import torch

def normalize(image):
    image = image / 255
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)

    return image

def denormalize(image, x, y):
    image = ((image / 2 + 0.5).clamp(0, 1))
    image = transforms.Resize((y, x))(image)

    image = image.cpu().permute(0, 2, 3, 1).numpy()

    return (image * 255).round().astype("uint8")

def denormalize_for_norm(image):
    image = (image/2+0.5).clamp(0, 1)

    return image

import cv2

import os
import time
import random

def import_LUT(LUTs_path):
    LUTs = []
    
    for filename in os.listdir(LUTs_path):
        if filename.endswith(".cube") or filename.endswith(".CUBE"):
            LUT_path = os.path.join(LUTs_path, filename)
            
            with open(LUT_path, 'r') as file:
                lines = file.readlines()
         
                for line in lines:
                    if line.startswith('LUT_3D_SIZE'):
                        lut_size = int(line.split()[-1]) 
                        break
                
                current_LUT = []

                for line in lines:
                    if line.startswith('#') or line.startswith('LUT_3D_SIZE') or line.strip() == "":
                        continue

                    if "TITLE" in line or "DOMAIN_MIN" in line or "DOMAIN_MAX" in line:
                        continue

                    values = line.split()
                    if len(values) == 3: 
                        current_LUT.append([float(val) for val in values])

                if len(current_LUT) != 32*32*32:
                    continue

                LUT_array = np.array(current_LUT, dtype=np.float32)
                LUTs.append(LUT_array)
    return LUTs

class ApplyLUT:
    def __init__(self, LUTs):
        self.LUTs = LUTs
        self.lut_size = 32

    def __call__(self, image):
        random.seed(time.time())

        LUT = random.choice(self.LUTs)
        LUT_filter = ImageFilter.Color3DLUT(self.lut_size, LUT)

        image = Image.fromarray(((image.permute(1, 2, 0).numpy()) * 255).astype(np.uint8))

        transformed_image = image.filter(LUT_filter)
        transformed_image = np.array(transformed_image) / 255.0
        transformed_image = torch.tensor(transformed_image, dtype=torch.float32).permute(2, 0, 1)
        
        return transformed_image
    
    def convert_LUT_to_3d(self, LUT):
        LUT_3d = np.zeros((self.lut_size, self.lut_size, self.lut_size, 3), dtype=np.float32)
    
        idx = 0
    
        for i in range(self.lut_size):
            for j in range(self.lut_size):
                for k in range(self.lut_size):
                    if len(LUT[idx]) == 3: 
                        r, g, b = LUT[idx]
                        LUT_3d[i, j, k] = [r, g, b]
                        idx += 1
  
        LUT_3d = np.transpose(LUT_3d, (3, 0, 1, 2))

        return LUT_3d

from PIL import Image, ImageFilter

import numpy as np

from torch.utils.data import Dataset

import torchvision.transforms as transforms

def preprocess(image_path, w, h):
    image = Image.open(image_path).convert("RGB")
    x, y = image.size

    image = transforms.Resize((256,256))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)

    image = np.array(image).astype(np.float32)
    image = normalize(image)
    
    return image , x, y

class CustomDataset(Dataset):
    def __init__(self, content_image_path, style_image_path, height = 256, width = 256):
        self.x = None
        self.y = None
        self.content_image_paths = content_image_path
        self.style_image_paths = style_image_path
        self.height = height
        self.width = width
    
    def __len__(self):
        return len(self.content_image_paths)

    def __getitem__(self, idx):
        content_image, self.x, self.y = preprocess(self.content_image_paths[idx], self.height, self.width)
        style_image, _, _ = preprocess(self.style_image_paths[idx], self.height, self.width)

        content_image = 2*content_image-1
        style_image = 2*style_image-1
        
        return content_image, style_image, self.x, self.y

# OPs for saving image from raw data

def save_image(image, filename):
    image = Image.fromarray(image)
    image.save(filename)

# recorde using tensorboard

def recode(writer, images, loss, epoch, batch, mode):
    writer.add_scalar(f"{mode}_L_con_epoch_{epoch}", loss['L_con'], batch)
    writer.add_scalar(f"{mode}_L_rec_epoch_{epoch}", loss['L_rec'], batch)
    writer.add_scalar(f"{mode}_Loss_epoch_{epoch}", loss['Loss'], batch)

    writer.add_image(f"{mode}_color_style_transfer epoch:{epoch}, batch:{batch}", images)

    return

# save weight

def save_weight(DNCM_Encoder_weight, epoch, batch):
    torch.save(DNCM_Encoder_weight, f'/data/lhayoung9/repos/color_style_transfer/DNCM_Encoder_epoch_{epoch}_batch_{batch}.pth')
    return