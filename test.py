import os
import numpy as np
from data.data_loader import CIFAR10Dataloader
from src.loss import CrossEntropyLoss
from src.model import MLPModel
# 获取项目根目录（假设该文件在项目根目录下）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

dataloader_kwargs = {
    "data_dir": os.path.join(PROJECT_ROOT, "data/pre"),
    "batch_size": 32,
}


ckpt_path = "models/model_epoch_100.pkl"


def main():
    dataloader = CIFAR10Dataloader(** dataloader_kwargs)
    model = MLPModel()
    model.load_model_dict(ckpt_path)  # 从已经训练好的权重加载模型
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for x_batch, y_batch in dataloader.generate_test_batch():
        y_pred = model.forward(x_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(x_batch)

    test_loss = total_loss / len(dataloader.y_test)
    test_acc = total_acc / len(dataloader.y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Checkpoint: {ckpt_path} | ")


if __name__ == "__main__":
    main()
