import argparse
import logging
import os
import tensorflow as tf
import keras.callbacks
from keras import backend as K
from keras import utils
from model.network import Glcic
from model.config import Config
from model import util
import dataset


class PrintAccuracy(keras.callbacks.Callback):
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def on_batch_end(self, batch, logs={}):
        self.logger.info("accuracy: %s", logs.get('acc'))


FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("---start---")

# GPUのメモリは必要な分だけ確保する。
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)
# set_debugger_session()

config = Config()

argparser = argparse.ArgumentParser(
    description="Globally and Locally Consistent Image Completion(GLCIC)"
    + " - train model.")
argparser.add_argument('--data_dir', type=str,
                       required=True, help="データセット配置ディレクトリ." +
                       "data_dir/train, data_dir/valの両方がある想定.")
argparser.add_argument('--stage', type=int,
                       required=True,
                       help="トレーニングステージ.1:generator only, " +
                       "2:discriminator only, 3:all",
                       choices=[1, 2, 3])
argparser.add_argument('--weights_path', type=str,
                       required=False, help="モデルの重みファイルのパス")
args = argparser.parse_args()
logger.info("args: %s", args)

# GPU数やバッチ数を指定。複数GPUでの動作は未検証
config.gpu_num = 1
config.batch_size = 2
util.out_name_pattern = "(.+_loss$|.+_debug$)"

# 学習モデル
network = Glcic(batch_size=config.batch_size, input_shape=config.input_shape,
                mask_shape=config.mask_shape)

train_generator = True
train_discriminator = True
if args.stage == 1:
    # generatorのみ訓練
    model, base_model = network.compile_generator(
        gpu_num=config.gpu_num,
        learning_rate=config.learning_rate)
    train_discriminator = False
    steps_per_epoch = 100
    epochs = 100  # batch_size(16) * 100 * 100 iterations per stage
elif args.stage == 2:
    # discriminatorのみ訓練
    model, base_model = network.compile_all(
        fix_generator_weight=True,
        gpu_num=config.gpu_num,
        learning_rate=config.learning_rate)
    train_generator = False
    steps_per_epoch = 100
    epochs = 100
elif args.stage == 3:
    # 相互訓練
    model, base_model = network.compile_all(
        fix_generator_weight=False,
        gpu_num=config.gpu_num,
        learning_rate=config.learning_rate,
        d_loss_alpha=config.d_loss_alpha)
    steps_per_epoch = 100
    epochs = 100

logger.info("train_generator:%s, train_discriminator:%s, "
            + "steps_per_epoch:%s, epochs:%s",
            train_generator, train_discriminator, steps_per_epoch, epochs)
logger.info(model.summary())

if args.weights_path:
    # 重みがあればロード
    logger.info("load weight:%s", args.weights_path)
    model.load_weights(args.weights_path, by_name=True)

# ネットワーク構成を画像として保存
utils.plot_model(model, './model.png', True, True)

# レイヤの重み更新有無を確認
for i, layer in enumerate(model.layers):
    if layer.__class__.__name__ == 'TimeDistributed':
        name = layer.layer.name
        trainable = layer.layer.trainable
    else:
        name = layer.name
        trainable = layer.trainable
    logger.info('%s %s:%s', i, name, trainable)

# 学習、検証データ生成器の準備
train_data_generator = dataset.DataGenerator(config).generate(
    os.path.join(args.data_dir, "train"), train_generator, train_discriminator)
val_data_generator = dataset.DataGenerator(config).generate(
    os.path.join(args.data_dir, "val"), train_generator, train_discriminator)

# コールバック準備
model_file_path = './nnmodel/glcic-stage{}-{}'.format(
    args.stage, '{epoch:02d}-{val_loss:.2f}.h5')
callbacks = [keras.callbacks.TerminateOnNaN(),
             keras.callbacks.TensorBoard(log_dir='./tb_log',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False),
             # keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
             #                                   verbose=1,
             #                                   factor=0.7,
             #                                   patience=10,
             #                                   min_lr=config.learning_rate
             #                                   / 30)
             keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                             verbose=1,
                                             save_weights_only=True,
                                             save_best_only=False,
                                             period=20)]

# 訓練
model.fit_generator(train_data_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    workers=4,
                    max_queue_size=10,
                    use_multiprocessing=True,
                    callbacks=callbacks,
                    validation_data=val_data_generator,
                    validation_steps=5)

# 訓練終了時の重みも保存
model_file_path = './nnmodel/glcic-latest-stage{}-{}'.format(
    args.stage, '{epoch:02d}-{val_loss:.2f}.h5')
model.save_weights(model_file_path)
