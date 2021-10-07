from lib import *


def make_datapath_list(phase='training_set'):
    root_path = '../cat_dog/data/training_set/'
    target_path = os.path.join(root_path + phase + '/**/*.jpg')
    # print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

#path_list = make_datapath_list('training_set')
#print(len(path_list))

def train_model(model, dataloader_dict, criterion, optimizer, device, writer, epoch):
    model = model.train()

    epoch_loss = 0.0
    correct_prediction = 0
    global epoch_accuracy

    for images, labels in tqdm(dataloader_dict):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        writer.add_scalar('Train Loss/Epoch', loss, epoch)
        _, preds = torch.max(outputs, 1)

        epoch_loss += loss.item() * images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_prediction += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(dataloader_dict.dataset)
        epoch_accuracy = correct_prediction.double() / len(dataloader_dict.dataset)

    return epoch_accuracy, epoch_loss


def evaluate_epoch(model, dataloader_dict, criterion,  device, writer, epoch):
    model.eval()

    epoch_loss = 0.0
    correct_prediction = 0
    global epoch_accuracy

    with torch.no_grad():
        for images, labels in tqdm(dataloader_dict):
            # Load images, labels to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            writer.add_scalar('Valid Loss/Epoch', loss, epoch)

            _, preds = torch.max(outputs, 1)

            epoch_loss += loss.item() * images.size(0)
            correct_prediction += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict.dataset)
            epoch_accuracy = correct_prediction.double() / len(dataloader_dict.dataset)

        return epoch_accuracy, epoch_loss

def update_param(model):
    param_to_update_1 = []
    param_to_update_2 = []

    update_param_name_1 = ['layer1']
    update_param_name_2 = ['fc.0.weight', 'fc.0.bias']

    for name, param in model.named_parameters():
        if name in update_param_name_1:
            param.requires_grad = True
            param_to_update_1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            param_to_update_2.append(param)
        else:
            param.requires_grad = False

    return param_to_update_1, param_to_update_2


def load_model(model, model_path):
    load_weights = torch.load(model_path, map_location={"cuda:0": "cpu"})
    model.load_state_dict(load_weights)

    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    return model