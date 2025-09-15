from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms

class ScannetDatasetRGB(Dataset):
    def __init__(self, cfg, split, ):
        self.data_root_ = cfg.get('data_root', '')
        
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(1024, scale=(0.85, 1.0)),  # 随机裁剪+缩放
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
        ])
        self.transform = transform

        if split == 'train':
            rgb_data_dir = os.path.join(self.data_root_, 'tasks', 'scannet_frames_25k')
            episode_names = [name for name in os.listdir(rgb_data_dir)]
            self.image_files_dir = [os.path.join(rgb_data_dir, episode, 'color') for episode in episode_names]
            self.image_files = [os.path.join(dir, img) for dir in self.image_files_dir for img in os.listdir(dir) if img.endswith('.jpg') or img.endswith('.png')]
        elif split == 'val':
            rgb_data_dir = os.path.join(self.data_root_, 'tasks', 'scannet_frames_test')
            episode_names = [name for name in os.listdir(rgb_data_dir) if os.path.isdir(os.path.join(rgb_data_dir, name))]
            self.image_files_dir = [os.path.join(rgb_data_dir, episode, 'color') for episode in episode_names]
            self.image_files = [os.path.join(dir, img) for dir in self.image_files_dir for img in os.listdir(dir) if img.endswith('.jpg') or img.endswith('.png')]
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    from torchvision import transforms
    cfg = OmegaConf.load('config/radio_enc_ms_dec_config.yaml')
    
    dataset = ScannetDatasetRGB(cfg, cfg.mode)
    print(f"Number of images in the dataset: {len(dataset)}")
    for i in range(5):
        img = dataset[i]
        print(f"Image {i}, tensor shape: {img.shape}")
