from keras import backend as K
import tensorflow as tf
from model import util


def discriminator(alpha):
    def _f(true, pred):
        """discriminatorの損失関数
            正解データ、generatorにより作成されたデータをまとめて評価する
            Args:
                true:
                    predと同じ次元であればなんでも良い。
                    Kerasのfunctional apiで使いたいのでこのIFにする。
                pred: [real, fake]
                    sigmoidの結果が0or1に2極化してしまうため、ネットワークの出力はそのままとし、この損失関数内でsigmoid_cross_entropy_with_logitsで処理する。
                    　参考とする実装に合わせる。
    　                https://github.com/tadax/glcic
        """
        real = pred[:, 0]
        fake = pred[:, 1]
        real = util.tfprint(real, "discriminator_real_debug")
        fake = util.tfprint(fake, "discriminator_fake_debug")
        loss_real = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real, labels=tf.ones_like(real))) * alpha
        # loss_real = util.tfprint(loss_real, "loss_real_debug")
        loss_fake = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake, labels=tf.zeros_like(fake))) * alpha
        # loss_fake = util.tfprint(loss_fake, "loss_fake_debug")
        loss = loss_real + loss_fake
        loss = util.tfprint(loss, "discriminator_loss")
        return loss
        # assert_op = tf.Assert(K.all([pred >= 0, pred <= 1]), [pred])
        # with tf.control_dependencies([assert_op]):
        #     # predが1または0の場合にlossが-infとなる事を避けるための微少値
        #     # discriminatorの出力は真贋の確率を示す0~1であるべき
        #     d = 1e-7
        #     pred = util.tfprint(pred, "discriminator_pred_debug")
        #     pred = tf.clip_by_value(pred, d, 1 - d)
        #     pred = util.tfprint(pred, "discriminator_clipped_pred_debug")
        #     real = pred[:, 0]
        #     fake = pred[:, 1]
        #     real = util.tfprint(real, "discriminator_real_debug")
        #     fake = util.tfprint(fake, "discriminator_fake_debug")
        #     # 重み
        #     # 論文には以下の用にあるので0.0004とする。
        #     #   我々は、Places2データセットから得られた8つ、097、967のトレーニング画像を使用して
        #     #   モデルを訓練する[Zhou et al。 2016]。このデータセットには、多様なシーンの画像が
        #     #   含まれており、もともとはシーンの分類に使用されていました。重み付けのハイパーパラメータ
        #     #   をα= 0.0004に設定し、96画像のバッチサイズを使用して訓練する。完了ネットワークは、
        #     #   TC = 90,000反復で訓練される。ディスクリミネータはTD = 10,000反復でトレーニング
        #     #   されます。最終的には両方ともトレイン= 50万回のトータルに達するように共同で訓練されて
        #     #   います。トレーニングの手順は、4台のK80 GPUを搭載した1台のマシンで約2ヶ月かかります。
        #     alpha = 0.0004
        #     # real_loss = -alpha * K.mean(K.log(real))
        #     real_loss = -alpha * K.mean(K.binary_crossentropy(
        #         K.ones_like(real), real))
        #     real_loss = util.tfprint(real_loss, "real_loss")
        #     # fake_loss = -alpha * K.mean(K.log(1 - fake))
        #     fake_loss = -alpha * K.mean(K.binary_crossentropy(
        #         K.zeros_like(fake), fake))
        #     fake_loss = util.tfprint(fake_loss, "fake_loss")
        #     loss = real_loss + fake_loss
        #     loss = util.tfprint(loss, "discriminator_loss")
        #     return loss
    return _f
