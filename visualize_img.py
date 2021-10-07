import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_dir = '/home/huynth/PycharmProjects/cat_dog/data/training_set/training_set'

def show_img(img_list, nrows, ncols, type_pet):
    fig = plt.figure(figsize=(8,8))
    for i, img_path in enumerate(img_list):
        img = mpimg.imread(os.path.join(train_dir, type_pet) + img_path)
        fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    img = os.listdir(os.path.join(train_dir, 'dogs/'))
    show_img(img[:6], 2, 3, 'dogs/')