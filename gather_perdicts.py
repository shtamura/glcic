import os
import glob
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = sys.argv[1]
if not os.path.isdir(root):
    print('{} is not directory.'.format(root))
    sys.exit()

epoch20 = 'stage3-20'
other_epochs = ['stage3-40', 'stage3-60', 'stage3-80', 'stage3-100']

paths20 = glob.glob(os.path.join(root, epoch20, '*.png'))
# predictの出力画像中、左側の画像の位置と各画像間の間隔
x = 76
y = 85
w = 136
h = 135
span = 28
images = []
filenames = []
for path in paths20:
    images_per_epoch = []
    # stage20のファイルからmaskされた画像とGTを取得
    # 入力画像
    img = np.array(Image.open(path))
    images_per_epoch.append(img[y:y + h, x:x + w, :])
    # GT
    x_gt = x + 2 * (span + w)
    images_per_epoch.append(img[y:y + h, x_gt:x_gt + w, :])
    # 予測結果
    x_predict = x + (span + w)
    images_per_epoch.append(img[y:y + h, x_predict:x_predict + w, :])

    filename = re.split('/', path)[-1]
    filenames.append(filename)
    # 他のepochの画像群にある同名ファイルから予測結果部分を切り出す
    for epoch in other_epochs:
        path = os.path.join(root, epoch, filename)
        img = np.array(Image.open(path))
        images_per_epoch.append(img[y:y + h, x_predict:x_predict + w, :])

    images.append(images_per_epoch)

# 入力画像毎に結果をまとめる
for image, filename in zip(images, filenames):
    print("processing {}.".format(filename))

    def plot_image(_img, _label, _num):
        plt.subplot(1, 7, _num)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.imshow(_img)
        # plt.axis('off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.xlabel(_label)

    plt.figure(figsize=(25, 4))
    plot_image(image[0], 'Input', 1)
    plot_image(image[1], 'Ground Truth', 2)
    plot_image(image[2], '20epochs', 3)
    plot_image(image[3], '40epochs', 4)
    plot_image(image[4], '60epochs', 5)
    plot_image(image[5], '80epochs', 6)
    plot_image(image[6], '100epochs', 7)
    plt.savefig('./out/merged_' + filename)
