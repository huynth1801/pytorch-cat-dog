from lib import *

class Image_Transform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'training_set': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test_set': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


if __name__=='__main__':
    img_file_path = '/home/huynth/PycharmProjects/cat_dog/data/training_set/training_set/dogs/dog.10.jpg'
    img = Image.open(img_file_path)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = Image_Transform(resize, mean, std)
    img_transformed = transform(img, phase='training_set')

    # (channel, height, width) -> (height, width, channel)
    img_transformed = img_transformed.permute(1, 2, 0)
    plt.imshow(img_transformed)
    plt.axis('off')
    plt.show()