import json

from src.GridSearcher import GridSearcher

hyper_param_defaults = {
    "input_dim": 3072,
    "hidden_size_1": 1024,
    "hidden_size_2": 256,
    "output_dim": 10,
    "activation_1": "relu",
    "activation_2": "relu",
    "activation_3": "softmax",
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}  # 超参数默认值（主要是神经网络结构和优化器参数）

dataloader_kwargs = {
    "data_dir": "data/pre",
    "batch_size": 32,
}  # 数据加载器参数（包括数据集路径、验证集大小、batch_size）

trainer_kwargs = {
    "n_epochs":40,
    "eval_step":5,  # 搜索超参数组合不需要在训练过程中评估
}  # 训练器参数（包括训练轮数、评估步数）


def main():
    hyper_param_opts = {
        "hidden_size_1": [1024,512],
        "hidden_size_2": [256,128],
        "lr": [0.05, 0.01],
        "ld": [0.001, 0.0005],
    }
    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults)
    results = searcher.search(dataloader_kwargs, trainer_kwargs, metric="acc")
    with open("gridsearch_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
