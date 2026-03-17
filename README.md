# Neuron-RCES

整理后的项目以 [main.py](/c:/Neuron-RCES/Neuron-RCES/main.py) 作为唯一训练入口，保留了当前实验流程需要的核心模块：

- `main.py`: Neuron-RCES 主训练与神经元级 MRC 选择流程
- `model.py`: 模型创建
- `dataloader.py`: 数据集与 DataLoader
- `utils.py`: 评估、日志与插值工具
- `optimizer.py`: 优化器与学习率调度
- `loss_utils.py`: 对比学习损失
- `models/`: 当前入口实际支持的模型实现

示例：

```bash
python main.py --model ResNet18 --dataset CIFAR10 --cal_neuron_mrc
python main.py --model ResNet18 --dataset CIFAR10 --epochs 10 --neurons_per_layer 1
```
