# Globally and Locally Consistent Image Completion (GLCIC)
画像の部分的な欠損を補完する手法である GLCIC を Keras で実装しています。   

![](resource/network-summary.png)
論文の表2より抜粋[^1]  

# 環境
- Python 3
- TensorFlow(TensorFlow-gpu)
- Keras
- OpenCV

# 学習に利用したデータセット
Places  
http://places2.csail.mit.edu/download.html  
```
mkdir /path/to/dataset
cd /path/to/dataset
wget http://data.csail.mit.edu/places/places365/test_256.tar
wget http://data.csail.mit.edu/places/places365/val_256.tar
unzip test_256.tar
unzip val_256.tar
# train_256.tarは21Gbyteと大きいためtest_256で代用した
mv test train
```

# 論文と異なる点、制約事項
- 学習効率優先のため、入力画像サイズを256*256に固定。
- generatorの出力が安定しない(ほぼノイズ...)ため、以下を調整。[^2]
  - generator, discriminatorともreluをleakyreluに変更。
  - generatorの出力層の活性化関数をsigmoidからtanhに変更。
- 論文中にある後処理(fast marching method, followed by Poisson image blending)は未実装。

# 使い方
## 学習
3ステージに分けて学習する。
- stage1  
generatorのみの学習。
```
python3 train.py --data_path /path/to/dataset --stage 1
```
- stage2  
discriminatorのみの学習。
```
python3 train_mrcnn.py --weights_path ./model/glcic-latest-stage1-xx.h5 --data_path /path/to/dataset --stage 2
```
- stage3  
RPN+Headの学習。
```
python3 train_mrcnn.py --weights_path ./model/glcic-latest-stage2-xx.h5 --data_path /path/to/dataset --stage 3
```
各ステージのイテレーションは少なめなので、実行環境や許容されるコストに合わせて調整してください。


## テスト
```
python3 predict.py --weights_path ./model/glcic-stage3-xx.h5 --input_path /path/to/testdata
```

### 結果
学習中...
- stage1: xxイテレーション
- stage2: xxイテレーション
- stage3: xxイテレーション


# 課題
学習中...


# 参考資料
- http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/ja/
- http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
- https://github.com/tadax/glcic
- https://qiita.com/underfitting/items/a0cbb035568dea33b2d7

[^1]: http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
[^2]: https://qiita.com/underfitting/items/a0cbb035568dea33b2d7
