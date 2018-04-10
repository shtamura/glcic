import argparse
import logging
import re
import sys

import cv2
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
resized_image, bin_mask, masked_image, mask_window = \
    gen.load_image(args.input_path)
if resized_image is None:
    logger.warn("指定の画像%sが存在しません.", args.input_path)
    sys.exit()

# 学習モデル
network = Glcic(batch_size=config.batch_size, input_shape=config.input_shape,
                mask_shape=config.mask_shape)
# コンパイル
# 補完画像を得られれば良いので、generatorのみを含むモデルを構築、コンパイルする。
model, _ = network.compile_generator(
    gpu_num=config.gpu_num,
    learning_rate=config.learning_rate)

# バッチ次元追加
in_masked_image = np.expand_dims(masked_image, 0)
in_bin_mask = np.expand_dims(bin_mask, 0)
# 予測
out_completion_image = \
    model.predict([in_masked_image, in_bin_mask], verbose=1,
                  batch_size=config.batch_size)
# バッチ次元削除
completion_image = np.squeeze(out_completion_image, 0)
# 非正規化（-1~1から0~255に戻す）
completion_image = gen.denormalize_image(completion_image)

# 入力画像、出力画像を保存
template = re.split('/|\.', args.input_path)[-2] + '_{}.png'
# 入力画像
cv2.imwrite(template.format('_in_res'), resized_image)
bin_mask = np.expand_dims(bin_mask, -1)
cv2.imwrite(template.format('_in_bin'), bin_mask * 255)
cv2.imwrite(template.format('_in_msk'), masked_image)

# 出力画像
cv2.imwrite(template.format('_out_raw'), completion_image)
# マスク部分のみ
cropped = completion_image * bin_mask
cv2.imwrite(template.format('_out_crp'), cropped)
# マスク部分を入力画像に重ねる
y1, x1, y2, x2 = mask_window
merged = resized_image.copy()
merged[y1:y2 + 1, x1:x2 + 1] = completion_image[y1:y2 + 1, x1:x2 + 1]
cv2.imwrite(template.format('_out_mrg'), merged)
