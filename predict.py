import argparse
import glob
import logging
import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from model.network import Glcic
from model.config import Config
from dataset import DataGenerator

"""
指定された画像の一部をランダムに切り抜いた画像を入力とする。
入力画像、補完結果画像を得る。
"""
FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("---start---")

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)
# set_debugger_session()

config = Config()

argparser = argparse.ArgumentParser(
    description="Globally and Locally Consistent Image Completion(GLCIC)"
    + " - generae image.")
argparser.add_argument('--input_path', type=str,
                       required=True, help="入力画像ファイルのパス." +
                       "[,]区切りで複数指定.ディレクトリ指定の場合は配下のファイルを全て読み込む." +
                       "data_dir/train, data_dir/valの両方がある想定.")
argparser.add_argument('--weight_path', type=str,
                       required=True, help="モデルの重みファイルのパス")
args = argparser.parse_args()
logger.info("args: %s", args)

config.gpu_num = 1
config.batch_size = 1

# 入力画像の読み込み
# リサイズ、ランダムに切り抜き
gen = DataGenerator(config)

# 学習モデル
network = Glcic(batch_size=config.batch_size, input_shape=config.input_shape,
                mask_shape=config.mask_shape)
# コンパイル
# 補完画像を得られれば良いので、generatorのみを含むモデルを構築、コンパイルする。
model, _ = network.compile_generator(
    gpu_num=config.gpu_num,
    learning_rate=config.learning_rate)


def pred(path):
    logger.info("input_path: %s", path)
    # 出力先パス
    template = './out/' + re.split('/|\.', path)[-2] + '_{}.png'

    # 入力画像
    resized_image, bin_mask, masked_image, mask_window = \
        gen.load_image(path)
    if resized_image is None:
        logger.warn("指定の画像%sが存在しません.", args.input_path)
        sys.exit()

    # cv2.imwrite(template.format('masked_image'), masked_image)
    # 入力イメージの正規化（0~255から-1~1へ）
    in_masked_image = gen.normalize_image(masked_image)

    # バッチ次元追加
    in_masked_image = np.expand_dims(in_masked_image, 0)
    in_bin_mask = np.expand_dims(bin_mask, 0)

    # 予測
    out_completion_image = \
        model.predict([in_masked_image, in_bin_mask], verbose=1,
                      batch_size=config.batch_size)
    # バッチ次元削除
    completion_image = np.squeeze(out_completion_image, 0)
    # 非正規化（-1~1から0~255に戻す）
    completion_image = gen.denormalize_image(completion_image)

    # 入力画像
    bin_mask = np.expand_dims(bin_mask, -1)
    # 出力画像
    # マスク部分のみ
    # cropped = completion_image * bin_mask
    # マスク部分を入力画像に重ねる
    # y1, x1, y2, x2 = mask_window
    # merged = resized_image.copy()
    # merged[y1:y2 + 1, x1:x2 + 1] = completion_image[y1:y2 + 1, x1:x2 + 1]
    merged = completion_image
    # cv2.imwrite(template.format('_in_res'), resized_image)
    # cv2.imwrite(template.format('_in_bin'), bin_mask * 255)
    # cv2.imwrite(template.format('_in_msk'), masked_image)
    # cv2.imwrite(template.format('_out_raw'), completion_image)
    # cv2.imwrite(template.format('_out_crp'), cropped)
    # cv2.imwrite(template.format('_out_mrg'), merged)

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

    def show_image(_img, _label, _num):
        plt.subplot(1, 3, _num)
        plt.imshow(_img)
        # plt.axis('off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.xlabel(_label)

    plt.figure(figsize=(6, 3))
    show_image(masked_image, 'Input', 1)
    show_image(merged, 'Output', 2)
    show_image(resized_image, 'Ground Trueth', 3)
    plt.savefig(template.format(''))


if os.path.isdir(args.input_path):
    paths = glob.glob(os.path.join(args.input_path, '*.jpg'))
else:
    paths = args.input_path.split(',')

for path in paths:
    pred(path)
