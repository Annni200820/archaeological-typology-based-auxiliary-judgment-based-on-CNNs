import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import label_map


# Custom dataset class
class CustomDataset(Dataset):
    def get_image_paths(self):
        return self.images

    def __init__(self, directory, labels_file, transform=None):
        self.directory = directory
        self.transform = transform
        # Read label file
        self.labels_df = pd.read_excel(labels_file)
        print(self.labels_df.columns)  # Print column names for verification
        # Filter image files in directory
        self.images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure 3-channel image
        if self.transform:
            image = self.transform(image)

        # Get filename (without path and extension)
        filename = os.path.basename(img_path).split('.png')[0]
        # Escape special characters in filename
        escaped_filename = re.escape(filename)
        # Find corresponding label in label file
        label_row = self.labels_df[
            self.labels_df.apply(lambda row: row.astype(str).str.contains(escaped_filename).any(), axis=1)]
        if not label_row.empty:
            label = label_row.iloc[0, 1]
            label = label_map[label]  # Map label to integer
        else:
            label = -1  # Set label to -1 if no matching filename is found
        return image, int(label)