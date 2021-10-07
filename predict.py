from utils import load_model
from lib import *
from torchvision.models import resnet50
from image_transform import Image_Transform

class_index = ['dogs', 'cats']

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_index[max_id]
        return predicted_label


predictor = Predictor(class_index)


def predict(img):
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    use_pretrained = True
    model = resnet50(pretrained=use_pretrained)
    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=2, bias=True),
                             nn.Sigmoid())
    model.eval()

    model = load_model(model, '/home/huynth/PycharmProjects/cat_dog/best_model.pth')

    # prepare inputdata
    '''
    img = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    '''
    transform = Image_Transform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze(0)

    output = model(img)
    response = predictor.predict_max(output)

    return response

if __name__== '__main__':
    test_img = Image.open('../cat_dog/test_img/pacto-visual-cWOzOnSoh6Q-unsplash.jpg').convert('RGB')
    plt.imshow(test_img)
    plt.show()
    print('\tPredicted image:' + predict(test_img))