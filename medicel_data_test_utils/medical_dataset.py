import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


# class dataset(Dataset):
#     def __init__(self, image_folder, csv_file, transform=None):
#         self.image_folder = image_folder
#         self.labels = pd.read_csv(csv_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels['label'].tolist())  # csv don't have column's name

#     def __getitem__(self, idx):
#         labelmap = {'l': 0, 'r': 1}
#         img_name = self.labels['figure'].tolist()[idx]  # figure
#         img_path = os.path.join(self.image_folder, img_name)
#         image = Image.open(img_path)
#         label = self.labels['label'].tolist()[idx]  # label
#         label = labelmap[label]

#         if self.transform:
#             image = self.transform(image)

#         return image, torch.tensor(label)

#class dataset(Dataset):
#    def __init__(self, image_folder, csv_file, transform=None):
#        self.image_folder = image_folder
#        self.labels = pd.read_csv(csv_file)
#        self.transform = transform

#   def __len__(self):
#        return len(self.labels['Class'].tolist())  # csv don't have column's name

#   def __getitem__(self, idx):
#        labelmap = {0: 0, 1: 1}
#        PatientID = self.labels['PatientID'].tolist()[idx]  # figure
#        StudyUID = self.labels['StudyUID'].tolist()[idx]  # figure
#        View = self.labels['View'].tolist()[idx]  # figure
#        Slice = self.labels['Slice'].tolist()[idx]  # figure
        
#       img_name = str(PatientID)+"_"+StudyUID+"_"+View+"_"+str(Slice)+"_grey.png"
        # print("img_name",img_name)

#       img_path = os.path.join(self.image_folder, img_name)
#       image = Image.open(img_path)
#       label = self.labels['Class'].tolist()[idx]  # label
#       label = labelmap[label]

#       if self.transform:
#           image = self.transform(image)

#       return image, torch.tensor(label)
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class dataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        #self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_folder) for f in filenames if f.endswith('.png')]
     #only extracted 40X images
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_folder) for f in filenames if f.endswith('.png') and 'SOB' in dp and '40X' in dp]
        print(f"Number of images extracted: {len(self.image_files)}")
        # Mapping for Tumor Class
        self.label_map = {'B': 0, 'M': 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        # Extract label (tumor class) from filename
        # Example filename: SOB_M_DC-14-12312-100-005.png
        tumor_class = img_path.split('/')[-1].split('_')[1]  # Extracts 'M' or 'B'
        label = self.label_map[tumor_class]  # Map the tumor class to a numeric label

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
