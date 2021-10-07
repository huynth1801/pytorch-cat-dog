import time
from image_transform import Image_Transform
from dataset import MyDataset
from lib import *
from utils import make_datapath_list, train_model, evaluate_epoch, update_param
from collections import defaultdict
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter

def main():
    history = defaultdict(list)
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 5
    best_val_acc = 0.0
    writer = SummaryWriter()

    train_list = make_datapath_list("training_set")
    val_list = make_datapath_list("test_set")

    #dataset
    train_ds = MyDataset(train_list, transform=Image_Transform(resize, mean, std), phase='training_set')
    val_ds = MyDataset(val_list, transform=Image_Transform(resize, mean, std), phase='test_set')

    # Dataloader
    batch_size = 16

    train_dataloader = DataLoader(train_ds, batch_size, shuffle=True)
    test_dataloader = DataLoader(val_ds, batch_size, shuffle=False)
    dataloader_dict = {"train": train_dataloader, 'test': test_dataloader}

    # Use resnet 50
    use_pretrained = True
    model = resnet50(pretrained=use_pretrained)
    # print(net)
    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=2, bias=True),
                             nn.Sigmoid())
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # optimizer
    params1, params2 = update_param(model)
    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 1e-3},
    ], momentum=0.9)

    for epoch in range(EPOCHS):
#        time.sleep(0.5)
        print(f'\nEpoch: [{epoch + 1}/{EPOCHS}]')
        print('-' * 40)

        train_acc, train_loss = train_model(model, dataloader_dict['train'], criterion, optimizer, \
                                             device, writer, epoch)
        val_acc, val_loss = evaluate_epoch(model, dataloader_dict['test'], criterion, \
                                            device, writer, epoch)

        print('Train Loss: {:.4f}\t Train Acc: {:.4f}'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}\t Val Acc: {:.4f}'.format(val_loss, val_acc))

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()