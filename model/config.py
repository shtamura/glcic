class Config:
    batch_size = 16
    gpu_num = 0
    learning_rate = 0.001  # kerasのAdamのデフォルト値
    # Discriminatorの損失関数で用いるパラメータ
    d_loss_alpha = 0.0004

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.input_shape = [value, value, 3]

    @property
    def mask_size(self):
        return self._mask_size

    @mask_size.setter
    def mask_size(self, value):
        self._mask_size = value
        self.mask_shape = [value, value, 3]
        # 欠損領域のサイズはマスクサイズの1/2~3/4とする
        # 論文では99以内とある。
        self.hole_min = value // 2
        self.hole_max = value // 4 * 3

    def __init__(self):
        self.input_size = 256
        self.mask_size = 128
