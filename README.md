# CIFAR-10图像分类
## Requirements

```bash
# NumPy is all you need ^^
pip install numpy

# Not exactly...
pip install matplotlib
pip install tqdm
```

## How to Run
### 模型训练

* 进入 [`train.py`](train.py) 修改以下部分（可选）：

  MLP神经网络结构参数：

  ```python
  # 自定义若干个线性层并添加到nn_architecture中
  # 指定每个线性层的输入维度、输出维度和后接的激活函数
  nn_architecture = [
      {"input_dim": 3072, "output_dim": 1024, "activation": "relu"},
      {"input_dim": 1024, "output_dim": 256, "activation": "relu"},
      {"input_dim": 256, "output_dim": 10, "activation": "softmax"},
  ]
  ```

  数据加载器参数：

  ```python
  dataloader_kwargs = {
      "data_dir": os.path.join(PROJECT_ROOT, "data/pre"),
      "batch_size": 32,
  }
  ```

  SGD优化器参数：

  ```python
  # 指定学习率、L2正则项系数、学习率衰减系数、学习率衰减步数
  optimizer_kwargs = {
      "lr": 0.05,
      "ld": 0.001,
      "decay_rate": 0.95,
      "decay_step": 6000,
  }
  ```

  训练器参数：

  ```python
  # 指定训练轮次、验证步数
  trainer_kwargs = {
      "n_epochs": 100,
      "eval_step": 10,
  }
  ```

* 进入仓库根目录，运行：

  ```bash
  python train.py
  ```

### 模型测试

* 将模型权重文件（一定要包括`.pkl`和`.json`文件）放至某一目录，例如`models/`

* 进入 [`test.py`](test.py) 修改以下部分（可选）：

  数据加载器参数：

  ```python
  # 在这里指定数据集所在路径、批量大小
  dataloader_kwargs = {
      "data_dir": os.path.join(PROJECT_ROOT, "data/pre"),
      "batch_size": 32,
  }
  ```

  模型权重文件的路径（指定`.pkl`的路径即可，`.json`文件会自动读取）：

  ```python
  ckpt_path = "models/model_epoch_100.pkl"
  ```

* 进入仓库根目录，运行：

  ```python
  python test.py
  ```

## Extra

### GridSearch

在这里进行超参数组合的搜索！

你可以在 [`gridsearch.py`](gridsearch.py) 中设置部分超参数的默认值和选项表（其中超参数名要严格和`nn_architecture`对应，譬如有多少个`hidden_size`和`activation`，并且`hidden_size`和`activation`后面必须跟上数字）

随后设置网格搜索超参数组合的基准（`loss`或者`acc`）以及其他参数（和前面模型训练、测试的部分的类似）

然后：

```python
python gridsearch.py
```

超参数组合的搜索结果会自动保存在 [`gridsearch_results.json`](gridsearch_results.json)

### ParamVis

[`utils/ParamVis.py`](ParamVis.py)提供了对模型网络初始化和训练后各层参数的可视化代码（包括直方图和热力图）

### 更多代码细节的说明详见报告
