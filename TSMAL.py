import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import time
import copy


# 1. 定义模型组件
class TSMA_Layer(nn.Module):
    """目标-阴影掩码辅助层"""

    def __init__(self, r=0.75):
        super().__init__()
        self.r = r  # 目标特征通道比例

    def forward(self, f_i, T_mask, S_mask):
        """
        f_i: 特征图 [B, C, H, W]
        T_mask: 目标掩码 [B, 1, H, W]
        S_mask: 阴影掩码 [B, 1, H, W]
        """
        # 通道分割
        B, C, H, W = f_i.shape
        c_t = int(C * self.r)
        f_t = f_i[:, :c_t]  # 目标特征通道
        f_s = f_i[:, c_t:]  # 阴影特征通道

        # 目标特征处理
        f_et = T_mask * f_t  # 增强目标区域特征
        f_st = (1 - T_mask) * f_t  # 抑制非目标区域特征

        # 阴影特征处理
        f_es = S_mask * f_s  # 增强阴影区域特征
        f_ss = (1 - S_mask) * f_s  # 抑制非阴影区域特征

        # 特征拼接
        f_e = torch.cat([f_et, f_es], dim=1)  # 增强特征
        f_s = torch.cat([f_st, f_ss], dim=1)  # 抑制特征

        return f_e, f_s


class MCA_Fusion(nn.Module):
    """多层坐标注意力融合模块 - 简化版"""

    def __init__(self, channels_list):
        super().__init__()
        self.channels_list = channels_list
        self.total_channels = sum(channels_list)

        # 使用1x1卷积进行特征融合
        self.conv = nn.Conv2d(self.total_channels, self.total_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.total_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features_list):
        # 统一特征尺寸 - 使用最后阶段的尺寸作为目标尺寸
        target_size = features_list[-1].shape[2:]

        # 调整所有特征图到相同尺寸
        resized_features = []
        for feat in features_list:
            # 使用插值调整尺寸
            resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized)

        # 通道拼接
        fused = torch.cat(resized_features, dim=1)

        # 应用卷积和激活
        fused = self.conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)

        return fused


class TSMAL(nn.Module):
    """完整的TSMAL网络架构"""

    def __init__(self, backbone='resnet18', r=0.75, lambda_val=0.8, num_classes=10):
        super().__init__()
        self.r = r
        self.lambda_val = lambda_val

        # 主干网络初始化
        if backbone == 'resnet18':
            base_model = resnet18(pretrained=False)

            # 修改第一层卷积以接受单通道输入
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # 移除最后的全连接层和池化层
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])

            # 分割为四个阶段并获取每个阶段的输出通道数
            self.stages = nn.ModuleList([
                nn.Sequential(*list(self.backbone.children())[:5]),  # stage1 (64 channels)
                nn.Sequential(*list(self.backbone.children())[5]),  # stage2 (128 channels)
                nn.Sequential(*list(self.backbone.children())[6]),  # stage3 (256 channels)
                nn.Sequential(*list(self.backbone.children())[7])  # stage4 (512 channels)
            ])

            # 每个阶段的输出通道数
            self.stage_channels = [64, 128, 256, 512]
            in_channels = 512
        else:
            raise ValueError(f"不支持的backbone类型: {backbone}")

        # TSMA层
        self.tsma_layers = nn.ModuleList([TSMA_Layer(r) for _ in range(len(self.stages))])

        # MCA融合模块 - 使用实际的阶段通道数
        self.mca_fusion = MCA_Fusion(self.stage_channels)

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sum(self.stage_channels), num_classes)  # 使用所有通道的总和
        )

    def forward(self, x, T_mask, S_mask):
        # 存储各阶段增强特征和抑制特征
        enhanced_features = []
        suppressed_features = []

        # 逐阶段处理
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 下采样掩码匹配特征图尺寸
            _, _, H, W = x.shape
            T_mask_down = F.interpolate(T_mask, size=(H, W), mode='bilinear', align_corners=False)
            S_mask_down = F.interpolate(S_mask, size=(H, W), mode='bilinear', align_corners=False)

            # TSMA层处理
            f_e, f_s = self.tsma_layers[i](x, T_mask_down, S_mask_down)
            enhanced_features.append(f_e)
            suppressed_features.append(f_s)
            x = f_e  # 将增强特征传递到下一层

        # MCA特征融合
        fused_features = self.mca_fusion(enhanced_features)

        # 分类预测
        pred = self.classifier(fused_features)

        # 背景抑制损失计算
        suppression_loss = 0
        for feat in suppressed_features:
            # 计算Frobenius范数的平方
            suppression_loss += torch.sum(feat ** 2)

        return pred, suppression_loss

    def compute_loss(self, pred, target, suppression_loss):
        # 分类损失
        ce_loss = F.cross_entropy(pred, target)

        # 总损失
        total_loss = self.lambda_val * ce_loss + (1 - self.lambda_val) * suppression_loss
        return total_loss


# 2. 修正后的数据预处理函数
def preprocess_masks(image, kappa=0.05, eta=0.04):
    """
    生成目标和阴影掩码
    image: 原始SAR图像 [C, H, W]
    kappa: 目标像素分位数 (top kappa%)
    eta: 阴影像素分位数 (bottom eta%)
    """
    # 确保图像是二维或三维张量
    if image.dim() == 3:  # [C, H, W]
        # 如果是单通道图像，压缩为二维
        if image.size(0) == 1:
            image = image.squeeze(0)  # [H, W]
        else:
            # 多通道图像，取平均值转为单通道
            image = torch.mean(image, dim=0)  # [H, W]
    elif image.dim() != 2:  # 如果不是二维
        raise ValueError(f"输入图像维度错误: {image.dim()}, 应为2或3维")

    # 目标掩码 (高亮度区域)
    sorted_vals, _ = torch.sort(image.view(-1))
    I_k = sorted_vals[-int(kappa * len(sorted_vals))]
    T_mask = (image > I_k).float()

    # 阴影掩码 (低亮度区域)
    I_eta = sorted_vals[int(eta * len(sorted_vals))]
    S_mask = (image < I_eta).float()

    # 形态学处理 (闭操作: 先膨胀后腐蚀)
    # 确保有批次和通道维度 [1, 1, H, W]
    T_mask = T_mask.unsqueeze(0).unsqueeze(0)
    S_mask = S_mask.unsqueeze(0).unsqueeze(0)

    # 闭操作 (膨胀 + 腐蚀)
    T_mask = F.max_pool2d(T_mask, kernel_size=3, padding=1, stride=1)
    T_mask = 1 - F.max_pool2d(1 - T_mask, kernel_size=3, padding=1, stride=1)

    S_mask = F.max_pool2d(S_mask, kernel_size=3, padding=1, stride=1)
    S_mask = 1 - F.max_pool2d(1 - S_mask, kernel_size=3, padding=1, stride=1)

    # 移除批次维度，保留通道维度 [1, H, W]
    T_mask = T_mask.squeeze(0)
    S_mask = S_mask.squeeze(0)

    return T_mask, S_mask


# 3. 自定义数据集类
class MSTARDataset(Dataset):
    def __init__(self, root_dir, transform=None, kappa=0.05, eta=0.04):
        """
        Args:
            root_dir (string): 数据集根目录
            transform (callable, optional): 应用于图像的变换
            kappa (float): 目标掩码参数
            eta (float): 阴影掩码参数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.kappa = kappa
        self.eta = eta

        # 获取所有类别和图像路径
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

        print(f"数据集加载完成，共 {len(self.image_paths)} 张图像，{len(self.classes)} 个类别")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像并转换为灰度
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        # 生成掩码
        T_mask, S_mask = preprocess_masks(image, self.kappa, self.eta)

        # 添加通道维度 [1, H, W]
        if T_mask.dim() == 2:
            T_mask = T_mask.unsqueeze(0)
            S_mask = S_mask.unsqueeze(0)

        return image, T_mask, S_mask, label


# 4. 训练和验证函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, T_mask, S_mask, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                T_mask = T_mask.to(device)
                S_mask = S_mask.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, suppression_loss = model(inputs, T_mask, S_mask)
                    loss = model.compute_loss(outputs, labels, suppression_loss)

                    _, preds = torch.max(outputs, 1)

                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# 5. 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 6. 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据集路径
    data_dir = "E:/论文/数据代码/代码/CAM_SAR_ATR_yolov8-main/MSTAR-10"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    # 检查数据集路径
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"错误: 数据集路径不存在！请检查路径: {train_dir} 和 {val_dir}")
        return

    # 数据变换
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ]),
    }

    # 创建数据集
    image_datasets = {
        'train': MSTARDataset(train_dir, transform=data_transforms['train']),
        'val': MSTARDataset(val_dir, transform=data_transforms['val'])
    }

    # 创建数据加载器
    batch_size = 16  # 减小批大小以节省内存
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # 减少工作线程数
            pin_memory=True
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # 减少工作线程数
            pin_memory=True
        )
    }

    # 检查数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['val']}")

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = TSMAL(backbone='resnet18', num_classes=len(image_datasets['train'].classes))
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练模型
    num_epochs = 100
    model, history = train_model(
        model,
        dataloaders,
        model.compute_loss,  # 使用模型自带的损失计算函数
        optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # 保存模型
    torch.save(model.state_dict(), 'tsmal_model.pth')
    print("模型已保存为 'tsmal_model.pth'")

    # 绘制训练历史
    plot_training_history(history)

    # 在验证集上最终评估
    model.eval()
    running_corrects = 0
    total = 0

    for inputs, T_mask, S_mask, labels in dataloaders['val']:
        inputs = inputs.to(device)
        T_mask = T_mask.to(device)
        S_mask = S_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs, T_mask, S_mask)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    final_acc = running_corrects.double() / total
    print(f'最终验证准确率: {final_acc:.4f}')


if __name__ == "__main__":
    main()