"""
对模型网络初始化和训练后各层参数的可视化
"""
import os
import sys
import json

sys.path.append("..")

import matplotlib.pyplot as plt

import src.layers as L
from src.model import MLPModel
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# 定义保存路径
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "images")
HISTOGRAM_PATH = os.path.join(IMAGE_ROOT, "histograms")
HEATMAP_PATH = os.path.join(IMAGE_ROOT, "heatmaps")

# 创建目录
os.makedirs(HISTOGRAM_PATH, exist_ok=True)
os.makedirs(HEATMAP_PATH, exist_ok=True)
ckpt_path = os.path.join(PROJECT_ROOT, "models/model_epoch_100.pkl")
nn_architecture = json.load(open(ckpt_path.replace(".pkl", ".json"), "r"))

model = MLPModel(nn_architecture)

# 初始化权重可视化
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        # 直方图保存路径
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution (Init)")
        plt.savefig(os.path.join(HISTOGRAM_PATH, f"layer_{i + 1}_init_hist.png"))
        plt.close()

        # 热力图保存路径
        plt.figure(figsize=(10, 6))
        plt.imshow(layer.W, cmap="hot", aspect="auto")
        plt.title(f"Layer {i + 1} Weight Matrix (Init)")
        plt.colorbar()
        plt.savefig(os.path.join(HEATMAP_PATH, f"layer_{i + 1}_init_heatmap.png"))
        plt.close()

# 加载训练后权重
model.load_model_dict(path=ckpt_path)

# 训练后权重可视化
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        # 直方图保存路径
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution (Trained)")
        plt.savefig(os.path.join(HISTOGRAM_PATH, f"layer_{i + 1}_trained_hist.png"))
        plt.close()

        # 热力图保存路径
        plt.figure(figsize=(10, 6))
        plt.imshow(layer.W, cmap="hot", aspect="auto")
        plt.title(f"Layer {i + 1} Weight Matrix (Trained)")
        plt.colorbar()
        plt.savefig(os.path.join(HEATMAP_PATH, f"layer_{i + 1}_trained_heatmap.png"))
        plt.close()