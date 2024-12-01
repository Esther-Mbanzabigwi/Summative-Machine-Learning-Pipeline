import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Custom Dataset class to handle a single image
class SingleImageDataset(Dataset):
    def __init__(self, image_path=None, transform=None, image=None):
        """
        Args:
            image_path (str): Path to the image.
            image: PIL Image
            transform (callable, optional): A function/transform to apply to the image.
        """
        self.image = image
        if image_path is not None:
            self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Single image dataset

    def __getitem__(self, idx):
        if self.image:
            image = self.image
        else:
            image = Image.open(self.image_path).convert("RGB")  # Ensure the image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image


def read_image(image_path=None, image=None):
    # Define the transformations
    data_mean = (0.485, 0.456, 0.406)
    data_std = (0.229, 0.224, 0.225)
    transformations = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

    # Create the dataset
    dataset = SingleImageDataset(image_path, 
                                transform=transformations,
                                image=image)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Iterate through the DataLoader
    return dataloader