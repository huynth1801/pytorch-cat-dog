from lib import *

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='training_set'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == 'training_set':
            label = img_path[42:46]
        elif self.phase == 'test_set':
            label = img_path[38:42]

        if label == 'dogs':
            label = 0
        elif label == 'cats':
            label = 1

        return img_transformed, label