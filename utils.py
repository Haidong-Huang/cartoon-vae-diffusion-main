import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch, torchvision
import random
import numpy as np
# from matplotlib import pyplot as plt

class UnlabeledImageFolder(Dataset):
    """Loads all images under root_dir (png/jpg) and applies a transform."""
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),                 # [0,1]
            T.Normalize([0.5]*3, [0.5]*3) # [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

class UnlabeledImageFolderDiffusion(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure no alpha channel
        if self.transform:
            image = self.transform(image)
        return image

# def plot_images(images):
#     plt.figure(figsize=(32, 32))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()


def save_images(images, path, **kwargs):
    kwargs.setdefault("nrow", 5)  # Set default nrow to 5 if not provided
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = UnlabeledImageFolderDiffusion(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def set_seed(seed: int):
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA),
    and turn on deterministic CuDNN to make runs reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # make CuDNN deterministic (may slow down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False