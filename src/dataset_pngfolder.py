from typing import List
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

class PngFolderDataset(Dataset):
    def __init__(self, roots: List[str], img_size=128):
        self.paths = []
        for r in roots:
            for p in glob.glob(os.path.join(r, "**", "*"), recursive=True):
                if p.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")) and "labels" not in p.lower():
                    self.paths.append(p)
        if not self.paths:
            raise RuntimeError(f"No images found under: {roots}")
        self.tx = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),  # [0,1], shape (1,H,W)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tx(img)
