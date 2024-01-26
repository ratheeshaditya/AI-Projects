import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class AnimeDataset(Dataset):
    """
    @AnimeDataset
    Parameters
    ---------------
    root_dir :- Root directory of the folder(i.e train, test, val) (string)
    transforms :- List of transformations to apply 
    x :- if "edge" returns edge image as input, else color.(string) 
    """
    def __init__(self, root_dir,transforms,x="edge"):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.x = x
        self.transforms = transforms


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        w = image.shape[1]//2
    
        input_image = image[:, :w, :]
        target_image = image[:, w:, :]
        #Apply the required transformations to the input and target image
        if self.transforms:
            input_image = self.transforms(input_image)
            target_image = self.transforms(target_image)

        if self.x=="edge": #if we want input as edge and target as full color 
            return target_image,input_image
        else: #if we want the the model to predict/convert the image to edges then this
            return input_image,target_image


if __name__ == "__main__":
    dataset = AnimeDataset(r"D:\My Projects\ML\Dataset\Anime\data\train")
    # print()
    loader = DataLoader(dataset, batch_size=5)
    # print(next(iter(loader)))
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()