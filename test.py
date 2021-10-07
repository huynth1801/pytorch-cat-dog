from image_transform import Image_Transform
from dataset import MyDataset
from lib import *
from utils import make_datapath_list, train_model, evaluate_epoch, update_param
from collections import defaultdict
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_list = make_datapath_list("training_set")
val_list = make_datapath_list("test_set")
#path_list = make_datapath_list('training_set')
#a = str(path_list[1])
#print(a[42:46])
# dataset
train_ds = MyDataset(train_list, transform=Image_Transform(resize, mean, std), phase='training_set')
val_ds = MyDataset(val_list, transform=Image_Transform(resize, mean, std), phase='test_set')
batch_size = 16

train_dataloader = DataLoader(train_ds, batch_size,shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size, shuffle=False)

dataloader_dict = {"train": train_dataloader, 'test': val_dataloader}

batch_iteration = iter(dataloader_dict['train'])
inputs, labels = next(batch_iteration)

print(inputs.size())
print(labels)