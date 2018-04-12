from logging import getLogger
import keras.layers as KL
import keras.layers.advanced_activations as KLA
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf

from model import loss
# from model import util

logger = getLogger(__name__)


class Glcic:
    """
        http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
        https://github.com/tadax/glcic
    """

    def __init__(self, batch_size, input_shape=[256, 256, 3],
                 mask_shape=[128, 128, 3]):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.mask_shape = mask_shape

    def _relu(self, input, name_prefix, leaky=True, trainable=True):
        if leaky:
            out = KLA.LeakyReLU(name='{}_leakyrelu'.format(name_prefix),
                                trainable=trainable)(input)
        else:
            out = KL.Activation('relu', name='{}_relu'.format(name_prefix),
                                trainable=trainable)(input)
        return out

    def _conv2d_part(self, name_prefix, input, filters, kernel_size, strides=1,
                     padding='same', trainable=True, leaky=True):
        out = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, kernel_initializer='he_uniform',
                        padding='same', name='{}'.format(name_prefix),
                        trainable=trainable)(input)
        out = KL.BatchNormalization(name='{}_bn'.format(name_prefix),
                                    trainable=trainable)(out)
        out = self._relu(out, name_prefix, leaky=leaky, trainable=trainable)
        return out

    def _dilated_conv2d_part(self, name_prefix, input, filters, kernel_size,
                             dilation_rate, padding='same', trainable=True,
                             leaky=True):
        out = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=1, kernel_initializer='he_uniform',
                        dilation_rate=dilation_rate, padding='same',
                        name='{}_dilated'.format(name_prefix),
                        trainable=trainable)(input)
        out = KL.BatchNormalization(name='{}_bn'.format(name_prefix),
                                    trainable=trainable)(out)
        out = self._relu(out, name_prefix, leaky=leaky, trainable=trainable)
        return out

    def _deconv2d_part(self, name_prefix, input, filters, kernel_size,
                       strides=1, padding='same', trainable=True, leaky=True):
        out = KL.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                 strides=strides,
                                 kernel_initializer='he_uniform',
                                 padding='same',
                                 name='{}_trans'.format(name_prefix),
                                 trainable=trainable)(input)
        out = KL.BatchNormalization(name='{}_bn'.format(name_prefix),
                                    trainable=trainable)(out)
        out = self._relu(out, name_prefix, leaky=leaky, trainable=trainable)
        return out

    def _build_model(self, inputs, outputs, name, trainable=True):
        with tf.device('/cpu:0'):
            model = Model(inputs, outputs, name)
            model.trainable = trainable
        return model

    def _generator(self, input_image, input_mask, trainable=True):
        """
            Returns:
                output layers:
        """

        out = self._conv2d_part('gen_conv1_1', input_image, filters=64,
                                kernel_size=5, strides=1,
                                trainable=trainable)

        # 特徴マップの高さ幅を半分に。
        out = self._conv2d_part('gen_conv2_1', out, filters=128,
                                kernel_size=3, strides=2,
                                trainable=trainable)
        out = self._conv2d_part('gen_conv2_2', out, filters=128,
                                kernel_size=3, strides=1,
                                trainable=trainable)
        # 特徴マップの高さ幅をさらに半分に。
        out = self._conv2d_part('gen_conv3_1', out, filters=256,
                                kernel_size=3, strides=2,
                                trainable=trainable)
        out = self._conv2d_part('gen_conv3_2', out, filters=256,
                                kernel_size=3, strides=1,
                                trainable=trainable)
        out = self._conv2d_part('gen_conv3_3', out, filters=256,
                                kernel_size=3, strides=1,
                                trainable=trainable)

        # 拡張畳み込み
        out = self._dilated_conv2d_part('gen_conv_3_4', out, filters=256,
                                        kernel_size=3, dilation_rate=2,
                                        trainable=trainable)

        out = self._dilated_conv2d_part('gen_conv_3_5', out, filters=256,
                                        kernel_size=3, dilation_rate=4,
                                        trainable=trainable)
        out = self._dilated_conv2d_part('gen_conv_3_6', out, filters=256,
                                        kernel_size=3, dilation_rate=8,
                                        trainable=trainable)
        out = self._dilated_conv2d_part('gen_conv_3_7', out, filters=256,
                                        kernel_size=3, dilation_rate=16,
                                        trainable=trainable)

        # 通常の畳込み
        out = self._conv2d_part('gen_conv3_8', out, filters=256,
                                kernel_size=3, strides=1,
                                trainable=trainable)
        out = self._conv2d_part('gen_conv3_9', out, filters=256,
                                kernel_size=3, strides=1,
                                trainable=trainable)

        # 逆畳み込みして元のサイズに戻していく
        # まずは2倍にする（元の1/2）
        out = self._deconv2d_part('gen_conv4_1', out, filters=128,
                                  kernel_size=4, strides=2,
                                  trainable=trainable)
        out = self._conv2d_part('gen_conv4_2', out, filters=128,
                                kernel_size=3, strides=1,
                                trainable=trainable)
        # 更に2倍（元のサイズ）
        out = self._deconv2d_part('gen_conv5_1', out, filters=64,
                                  kernel_size=4, strides=2,
                                  trainable=trainable)
        out = self._conv2d_part('gen_conv5_2', out, filters=32,
                                kernel_size=3, strides=1,
                                trainable=trainable)
        # 出力層はtanh
        # 勾配0とならないほうが学習が安定する。
        # https://qiita.com/underfitting/items/a0cbb035568dea33b2d7
        name_prefix = 'gen_conv5_3'
        out = KL.Conv2D(filters=3, kernel_size=3,
                        strides=1, kernel_initializer='glorot_uniform',
                        padding='same', name='{}'.format(name_prefix),
                        trainable=trainable)(out)
        out = KL.Activation('tanh',
                            name='{}_tanh'.format(name_prefix),
                            trainable=trainable)(out)

        # マスク領域で切り抜いて、正解データとマージする。
        # [N, 256, 256] → [N, 256,256,1]
        mask = KL.Reshape((self.input_shape[0], self.input_shape[1],
                           1), trainable=False)(input_mask)
        # x[0] * x[2] : out からmaskのビットが立っている領域を切り出す(マスク以外の領域を0にする)
        # x[1] * (1 - x[2]) : input_imageからmaskのビットが立っていない領域を切り出す
        # 上記２つをマージ（加算）することで、マスク部分のみNNの出力に置き換えた画像にする
        out = KL.Lambda(lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                        name='gen_merge_real',
                        trainable=False)([out, input_image, mask])

        model = self._build_model([input_image, input_mask], out, 'generator',
                                  trainable=trainable)
        return model

    def _global_discriminator(self, input):
        # input = KL.Input(
        #     shape=self.input_shape, name='gd_input', dtype='float32')
        out = self._conv2d_part('gd_conv1', input, filters=64,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('gd_conv2', out, filters=128,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('gd_conv3', out, filters=256,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('gd_conv4', out, filters=512,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('gd_conv5', out, filters=512,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('gd_conv6', out, filters=512,
                                kernel_size=5, strides=2, leaky=True)
        # out = KL.Flatten(name='gd_flatten7')(out)
        # Flatten()だとinput_shape不明と言われてflatにできなかったため、形状明示してreshape
        out = KL.Reshape((4 * 4 * 512,), name='gd_flatten7')(out)
        out = KL.Dense(1024, kernel_initializer='glorot_uniform',
                       name='gd_fc8')(out)

        return out

    def _local_discriminator(self, input):
        # input = KL.Input(
        #     shape=self.mask_shape, name='ld_input', dtype='float32')
        out = self._conv2d_part('ld_conv1', input, filters=64,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('ld_conv2', out, filters=128,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('ld_conv3', out, filters=256,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('ld_conv4', out, filters=512,
                                kernel_size=5, strides=2, leaky=True)
        out = self._conv2d_part('ld_conv5', out, filters=512,
                                kernel_size=5, strides=2, leaky=True)
        # out = KL.Flatten(name='ld_flatten6')(out)
        # Flatten()だとinput_shape不明と言われてflatにできなかったため、形状明示してreshape
        out = KL.Reshape((4 * 4 * 512,), name='ld_flatten6')(out)
        out = KL.Dense(1024, kernel_initializer='glorot_uniform',
                       name='ld_fc7')(out)

        return out

    def _discriminator(self, input_global, input_local):
        g_output = self._global_discriminator(input_global)
        l_output = self._local_discriminator(input_local)
        out = KL.Lambda(lambda x: K.concatenate(x),
                        name='d_concat1')([g_output, l_output])
        out = KL.Dense(1, kernel_initializer='glorot_uniform',
                       name='d_fc2')(out)
        # 論文中ではsigmoid -> cross_entropyだが、sigmoidした時点で0or1に偏り、損失が固定化されてしまう。
        # 勾配が早々に消失しているためと思われる。。。
        # 従って、ここではsigmoidせず、損失関数にてsigmoidを行う
        # out = KL.Activation('sigmoid', name='d_sigmoid3')(out)

        model = self._build_model(
            [input_global, input_local], out, 'discriminator')

        return model

    def _compile_model(self, model, losses, gpu_num, learning_rate):
        if gpu_num >= 2:
            # 複数GPU
            wrapped_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            wrapped_model = model

        wrapped_model.compile(loss=losses,
                              optimizer=Adam(lr=learning_rate))
        return wrapped_model

    def compile_generator(self, gpu_num=0, learning_rate=0.001):
        """generatorだけのネットワークを作成＆compileする
        """
        # マスク部分を一定の色（単純に黒にする）で塗りつぶした画像
        input_masked_image = KL.Input(
            shape=self.input_shape, name='input_masked_image', dtype='float32')
        # マスクは入力画像と同じサイズのバイナリマスク
        input_bin_mask = KL.Input(
            shape=self.input_shape[:2], name='input_bin_mask', dtype='float32')

        model = self._generator(input_masked_image, input_bin_mask)
        wrapped_model = self._compile_model(model, 'mean_squared_error',
                                            gpu_num, learning_rate)
        return wrapped_model, model

    def _crop_local(self, reals, fakes, mask_areas):
        """reals, fakes を mask_areasの領域 で切り抜いたデータを得る
        """
        # print("reals, fakes, masks:", reals, fakes, masks)
        # バッチ毎に分割して処理する
        fakes = tf.split(fakes, self.batch_size)
        reals = tf.split(reals, self.batch_size)
        mask_areas = tf.split(mask_areas, self.batch_size)
        real_locals = []
        fake_locals = []
        for real, fake, mask_area in zip(reals, fakes, mask_areas):
            # １次元目のバッチを示す次元を削除
            fake = K.squeeze(fake, 0)
            real = K.squeeze(real, 0)
            mask_area = K.cast(K.squeeze(mask_area, 0), tf.int32)
            top = mask_area[0]
            left = mask_area[1]
            h = mask_area[2] - top
            w = mask_area[3] - left

            # top = util.tfprint(top, prefix="top_debug")
            # left = util.tfprint(left, prefix="left_debug")
            # h = util.tfprint(h, prefix="h_debug")
            # w = util.tfprint(w, prefix="w_debug")

            fake_local = tf.image.crop_to_bounding_box(
                fake, top, left, h, w)
            fake_locals.append(fake_local)

            real_local = tf.image.crop_to_bounding_box(
                real, top, left, h, w)
            real_locals.append(real_local)

        fake_locals = K.stack(fake_locals)
        real_locals = K.stack(real_locals)
        # print("real_locals, fake_locals", real_locals, fake_locals)
        return [real_locals,  fake_locals]

    def compile_all(self, fix_generator_weight=False,
                    gpu_num=0, learning_rate=0.001, d_loss_alpha=0.0004):
        """generator + discriminator ネットワークを作成＆compileする
        """
        # マスク部分を一定の色（単純に黒にする）で塗りつぶした画像
        input_masked_image = KL.Input(
            shape=self.input_shape, name='input_masked_image', dtype='float32')
        # バイナリマスク
        input_bin_mask = KL.Input(
            shape=self.input_shape[:2], name='input_bin_mask', dtype='float32')
        # マスク領域
        # [y1,x1,y2,x2]
        input_mask_area = KL.Input(
            shape=[4], name='input_mask_area', dtype='int32')
        # 入力画像そのまま
        input_real_global = KL.Input(
            shape=self.input_shape, name='input_real_global', dtype='float32')
        input_real_local = KL.Input(
            shape=self.mask_shape, name='input_real_local', dtype='float32')

        model_gen = self._generator(input_masked_image, input_bin_mask,
                                    trainable=not fix_generator_weight)
        model_dis = self._discriminator(input_real_global, input_real_local)

        # fake_global = model_gen([input_masked_image, input_bin_mask])
        fake_global = model_gen.layers[-1].output
        # print("fake_global: ", fake_global)

        if fix_generator_weight:
            # generatorの重みは固定
            outputs = []
            losses = []
        else:
            outputs = [fake_global]
            losses = ['mean_squared_error']

        # 本物、generatorが生成した偽物をdiscriminatorに評価させる
        # localの画像はマスク領域の画像を切り抜いて評価
        real_local, fake_local = KL.Lambda(lambda x: self._crop_local(*x),
                                           name='crop_local')(
            [input_real_global, fake_global, input_mask_area])
        prob_real = model_dis([input_real_global, real_local])
        prob_fake = model_dis([fake_global, fake_local])
        # print("prob_real: ", prob_real)
        # print("prob_fake: ", prob_fake)

        # 判定結果をバッチ毎にまとめる。
        # [N, 2]の形式にする。
        def _stack(p_real, p_fake):
            # print("p_real: ", p_real)
            # print("p_fake: ", p_fake)
            # [[prob_real, prob_fake], ...] の形状にする
            prob = K.squeeze(K.stack([p_real, p_fake], -1), 1)
            return prob
        prob = KL.Lambda(lambda x: _stack(*x), name='stack_prob')(
            [prob_real, prob_fake])
        # print("prob: ", prob)
        outputs.append(prob)
        losses.append(loss.discriminator(d_loss_alpha))

        model_all = self._build_model([input_masked_image, input_bin_mask,
                                       input_mask_area, input_real_global],
                                      outputs, 'gen+dis')

        wrapped_model = self._compile_model(model_all, losses,
                                            gpu_num, learning_rate)
        return wrapped_model, model_all
