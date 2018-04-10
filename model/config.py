class Config:
    batch_size = 2
    gpu_num = 0
    input_size = 256
    mask_size = 128
    learning_rate = 0.001  # kerasのAdamのデフォルト値
    # Discriminatorの損失関数で用いるパラメータ
    d_loss_alpha = 0.0004

    def __init__(self):
        self.input_shape = [self.input_size, self.input_size, 3]
        self.mask_shape = [self.mask_size, self.mask_size, 3]
