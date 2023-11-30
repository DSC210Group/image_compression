from PIL import Image
import numpy as np
import os


class ImageDataset:
    def __init__(self, directory='./rotated_dataset/'):
        """
        Initialize the dataset with the path to the directory containing images.
        """
        self.directory = directory
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.file_names = [f for f in os.listdir(directory)]

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Load and return an image at the specified index.
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return np.array(image)

    def get_filename(self,index):
        """
        Return the filenames of the images.
        """
        return self.file_names[index]
    
    def get_paths(self):
        """
        Return the file paths of the images.
        """
        return self.image_paths
    