import os
import matplotlib.pyplot as plt
from data.data_loader import CIFAR10Dataloader
from src.loss import CrossEntropyLoss
from src.model import MLPModel
from src.optimizer import SGDOptimizer
from src.trainer import Trainer

# 获取项目根目录（假设该文件在项目根目录下）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 神经网络结构参数
nn_architecture = [
    {"input_dim": 3072, "output_dim": 1024, "activation": "relu"},
    {"input_dim": 1024, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 10, "activation": "softmax"},
]

# 数据加载器参数（使用绝对路径）
dataloader_kwargs = {
    "data_dir": os.path.join(PROJECT_ROOT, "data/pre"),
    "batch_size": 32,
}

# 优化器参数
optimizer_kwargs = {
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}

# 训练器参数（显式指定参数名称）
trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 5,
}


def main():
    # 创建必要的保存目录
    required_dirs = ["logs", "models", "images"]
    for dir_name in required_dirs:
        os.makedirs(os.path.join(PROJECT_ROOT, dir_name), exist_ok=True)

    # 初始化数据加载器
    dataloader = CIFAR10Dataloader(** dataloader_kwargs)

    # 初始化模型组件
    model = MLPModel(nn_architecture)
    optimizer = SGDOptimizer(** optimizer_kwargs)
    loss = CrossEntropyLoss()

    # 初始化训练器（显式传递所有必要参数）
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss=loss,
        dataloader=dataloader,
        n_epochs=trainer_kwargs["n_epochs"],
        eval_step=trainer_kwargs["eval_step"]
    )

    # 执行训练流程
    trainer.train(save_ckpt=True, verbose=True)

    # 保存训练结果
    trainer.save_log(os.path.join(PROJECT_ROOT, "logs/"))
    trainer.save_best_model(os.path.join(PROJECT_ROOT, "models/"),
                            metric="loss", n=3, keep_last=True)
    trainer.clear_cache()

    # 关闭所有绘图窗口
    plt.close('all')


if __name__ == "__main__":
    main()