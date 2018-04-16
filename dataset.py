import numpy as np
import cv2
import random
import os
import glob
import logging


logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self, config, random_hole=True):
        self.config = config
        self.random_hole = random_hole

    def normalize_image(self, image):
        """generatorの出力はtanhで活性化されていることを考慮し、-1~1に正規化する。
        """
        # 127.5=RGB値である255の半分
        return (image / 127.5) - 1

    def denormalize_image(self, image):
        """generatorの出力(-1~1)を元に0~255に戻す。
        """
        # 127.5=RGB値である255の半分
        return ((image + 1) * 127.5).astype(np.uint8)

    def load_image(self, path):
        """
        入力画像、バイナリマスク、バイナリマスクで切り出した画像を得る。
        マスクはランダム。複数の入力画像に共通で適用できるマスクであること。
        """
        logger.debug("load file:%s", path)

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warn("指定の画像が存在しない")
            return None, None, None, None

        resized_image, window, _ = \
            resize_with_padding(image,
                                self.config.input_size,
                                self.config.input_size)
        # mask
        y1, x1, y2, x2 = window
        if y2 - y1 < self.config.mask_size or x2 - x1 < self.config.mask_size:
            logger.warn("指定のマスク領域が確保出来ない画像なのでスキップ: %s", window)
            return None, None, None, None

        # マスク領域
        if self.random_hole:
            y1 = random.randint(y1, y2 - self.config.mask_size - 1)
            x1 = random.randint(x1, x2 - self.config.mask_size - 1)
        else:
            y1 = y1 + (y2 - y1) // 4
            x1 = x1 + (x2 - x1) // 4
        y2 = y1 + self.config.mask_size
        x2 = x1 + self.config.mask_size
        mask_window = (y1, x1, y2, x2)

        # マスク領域内の穴（マスクビットを立てる領域）
        if self.random_hole:
            h, w = np.random.randint(self.config.hole_min,
                                     self.config.hole_max + 1, 2)
            py1 = y1 + np.random.randint(0, self.config.mask_size - h)
            px1 = x1 + np.random.randint(0, self.config.mask_size - w)
        else:
            h, w = self.config.hole_max, self.config.hole_max
            py1 = y1 + (self.config.mask_size - h) // 2
            px1 = x1 + (self.config.mask_size - w) // 2
        py2 = py1 + h
        px2 = px1 + w

        masked_image = resized_image.copy()
        # 論文中では「データセット中の画像の平均ピクセル値で塗りつぶす」とあるが0にする。
        # ネットワークに投入する際の正規化で−1（非0）になるのでこれで良さそう。
        masked_image[py1:py2 + 1, px1:px2 + 1, :] = 0

        # バイナリマスク
        bin_mask = np.zeros(resized_image.shape[0:2])
        bin_mask[py1:py2 + 1, px1:px2 + 1] = 1

        return resized_image, bin_mask, masked_image, mask_window

    def generate(self, data_dir, train_generator=True,
                 train_discriminator=True):
        i = 0
        while True:
            paths = glob.glob(os.path.join(data_dir, '*.jpg'))
            # 並列でトレーニングする場合に各スレッドで異なる画像を利用するためshuffle
            random.shuffle(paths)
            for path in paths:
                if i == 0:
                    resized_images = []
                    bin_masks = []
                    masked_images = []
                    mask_windows = []

                resized_image, bin_mask, masked_image, mask_window = \
                    self.load_image(path)
                if resized_image is None:
                    continue

                i += 1
                # 正規化
                resized_image = self.normalize_image(resized_image)
                masked_image = self.normalize_image(masked_image)
                resized_images.append(resized_image)
                bin_masks.append(bin_mask)
                masked_images.append(masked_image)
                mask_windows.append(mask_window)

                if i == self.config.batch_size:
                    resized_images = np.array(resized_images)
                    bin_masks = np.array(bin_masks)
                    masked_images = np.array(masked_images)
                    mask_windows = np.array(mask_windows, dtype=np.int32)

                    inputs = [masked_images, bin_masks]
                    targets = []
                    if train_generator:
                        targets.append(resized_images)
                    if train_discriminator:
                        inputs.append(mask_windows)
                        inputs.append(resized_images)
                        # discriminatorの正解データ
                        # networkの実装に合わせて[real, fake]=[1,0]とする
                        targets.append(
                            np.tile([1, 0], (self.config.batch_size, 1)))
                    i = 0
                    yield inputs, targets


def resize_with_padding(image_array, min_size, max_size):
    """アスペクト比を維持したままリサイズする。
    高さ、または幅の小さい方がmin_sizeとなるようリサイズする。
    リサイズの結果、高さ、または幅の大きい方がmax_sizeを超える場合は、高さ、または幅の大きい方をmax_sizeとする。
    リサイズ後画像を max_size*max_size の枠の中央に配置し、周辺を0でPaddingする。

    Args:
        image_array: [h,w,3]の配列
        min_size:
        max_size:

    Returns:
        resized_image: リサイズ後の画像
        window: (y1, x1, y2, x2). リサイズ後の画像が画像全体のどの位置にあるかを示す座標
        scale: 元画像に対してのスケール
    """
    h, w = image_array.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    scale = max(1, min_size / min(h, w))

    # max_sizeを超えないよう調整
    image_max = max(h, w)
    if round(image_max * scale) > max_size:
        scale = max_size / image_max

    if scale != 1:
        image_array = cv2.resize(image_array, None, fx=scale, fy=scale)
    # Padding
    h, w = image_array.shape[:2]
    top_pad = (max_size - h) // 2
    bottom_pad = max_size - h - top_pad
    left_pad = (max_size - w) // 2
    right_pad = max_size - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image_array = np.pad(image_array, padding,
                         mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return image_array, window, scale


def show_images(config, paths):
    gen = DataGenerator(config)
    for path in paths:
        image, bin_mask, masked_image, _ = gen.load_image(path)
        print(bin_mask)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.imshow('masked_image', masked_image)
        cv2.waitKey(0)
