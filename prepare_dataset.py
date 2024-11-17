from torch.utils.data import  Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataTrain(Dataset):
    def __init__(self, inp, tar, transform):
        self.inp = inp
        self.tar = tar
        self.transform = transform

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, idx):
        inp = self.inp[idx]
        tar = self.tar[idx]
        inp = Image.open(inp).convert("RGB")
        tar = Image.open(tar).convert("RGB")
        
        image_a = self.transform(inp)
        image_b = self.transform(tar)
        return image_a, image_b


def create_data(dataroot: str):
    """
    Creates the `torch.utils.data.Datasets` for the appropriate dataroot.
    arg:
        dataroot (str): cityscapes or camvid
    """

    # Initialize the paths of train and val data
    if dataroot == 'cityscapes':
        X_TRAIN_DIR = 'cityscapes_data/train/'
        Y_TRAIN_DIR = 'cityscapes_data/train_labels/'
        X_VAL_DIR = 'cityscapes_data/val/'
        Y_VAL_DIR = 'cityscapes_data/val_labels/'
    elif dataroot == 'camvid':
        X_TRAIN_DIR = 'camvid_data/train/'
        Y_TRAIN_DIR = 'camvid_data/train_labels/'
        X_VAL_DIR = 'camvid_data/val/'
        Y_VAL_DIR = 'camvid_data/val_labels/'

    # Get all the image paths
    X_train = [X_TRAIN_DIR+x for x in os.listdir(X_TRAIN_DIR) if x.endswith('.png')]
    y_train = [Y_TRAIN_DIR+x for x in os.listdir(Y_TRAIN_DIR) if x.endswith('.png')]
    X_val = [X_VAL_DIR+x for x in os.listdir(X_VAL_DIR) if x.endswith('.png')]
    y_val = [Y_VAL_DIR+x for x in os.listdir(Y_VAL_DIR) if x.endswith('.png')]

    # Define tranforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # val_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])

    # Create the torch Datasets
    train_dataset = CustomDataTrain(inp=y_train, tar=X_train, transform=train_transform)
    val_dataset = CustomDataTrain(inp=y_val, tar=X_val, transform=train_transform)

    return train_dataset, val_dataset

def single_image_preprocess(img_path):
    """
    Applies resize(256, 256) and convert an image into tensor.
    arg:
        img (str): Image path
    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    return transform(img)