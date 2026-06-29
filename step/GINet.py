import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class GraphInteractionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphInteractionLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return F.relu(self.fc(x))

class GINet(nn.Module):
    def __init__(self, num_classes):
        super(GINet, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the average pool and fc layer
        self.gil = GraphInteractionLayer(2048, 512)
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.semantic_centroids = nn.Parameter(torch.randn(num_classes, 512))

    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.size()
        features = features.view(B, C, -1).permute(0, 2, 1)
        features = self.gil(features)
        features = features.permute(0, 2, 1).view(B, 512, H, W)
        logits = self.classifier(features)
        return logits

class SECrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(SECrossEntropyLoss, self).__init__()
        self.semantic_centroids = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, logits, labels):
        if logits.ndimension() == 4:
            logits = logits.squeeze(2).squeeze(2)
        assert logits.ndimension() == 2, "The shape of logits should be [N, C, 1, 1] or [N, C], but the logits dim is {}.".format(logits.ndimension())

        batch_size, num_classes = logits.shape
        se_label = torch.zeros([batch_size, num_classes], device=logits.device)
        
        for i in range(batch_size):
            hist = torch.histc(labels[i].float(), bins=num_classes, min=0, max=num_classes - 1)
            hist = hist.float() / hist.sum().float()
            se_label[i] = (hist > 0).float()

        loss = F.binary_cross_entropy_with_logits(logits, se_label)
        return loss

# 示例训练代码
def train(model, criterion, optimizer, dataloader, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples

# 示例用法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GINet(num_classes=21).to(device)
criterion = SECrossEntropyLoss(num_classes=21, feature_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 示例数据加载器
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些随机数据
images = torch.randn(100, 3, 224, 224)
labels = torch.randint(0, 21, (100, 1, 1, 1))
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 训练和评估
for epoch in range(10):
    train(model, criterion, optimizer, dataloader, device)
    accuracy = evaluate(model, dataloader, device)
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.4f}')



######################
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticContextLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(SemanticContextLoss, self).__init__()
        # Define a learnable semantic centroid for each category
        self.semantic_centroids = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, semantic_nodes, labels):
        # semantic_nodes: [N, D], where N is the number of nodes, D is the feature dimension
        # labels: [N, num_classes], one-hot encoded labels indicating the presence of each class

        # Calculate the scores using dot product and sigmoid activation
        scores = torch.sigmoid(torch.matmul(semantic_nodes, self.semantic_centroids.t()))

        # Compute the Binary Cross-Entropy loss
        loss = F.binary_cross_entropy(scores, labels.float())
        
        return loss

# Example usage
if __name__ == "__main__":
    num_classes = 20
    feature_dim = 512
    batch_size = 16

    # Randomly generated semantic nodes and labels for demonstration
    semantic_nodes = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 2, (batch_size, num_classes))

    sc_loss = SemanticContextLoss(num_classes, feature_dim)
    loss = sc_loss(semantic_nodes, labels)

    print("SC Loss:", loss.item())


class SemanticContextLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(SemanticContextLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.semantic_centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

    def forward(self, semantic_nodes, labels):
        # semantic_nodes: [batch_size, num_nodes, embedding_dim]
        # labels: [batch_size, num_classes]
        
        batch_size = semantic_nodes.size(0)
        num_nodes = semantic_nodes.size(1)
        
        # Expand semantic centers to match the batch size
        semantic_centers_expanded = self.semantic_centers.expand(batch_size, -1, -1)  # [batch_size, num_classes, embedding_dim]
        
        # Compute similarity scores vi for each node and class
        similarity_scores = torch.bmm(semantic_nodes, semantic_centers_expanded.transpose(1, 2))  # [batch_size, num_nodes, num_classes]
        similarity_scores = self.sigmoid(similarity_scores)  # Apply sigmoid activation
        
        # Flatten similarity scores and labels for BCE loss
        similarity_scores_flat = similarity_scores.view(batch_size * num_nodes, self.num_classes)  # [batch_size * num_nodes, num_classes]
        labels_flat = labels.unsqueeze(1).expand(batch_size, num_nodes, self.num_classes)
        labels_flat = labels_flat.contiguous().view(batch_size * num_nodes, self.num_classes)  # [batch_size * num_nodes, num_classes]
        
        # Compute SC loss using BCE loss
        sc_loss = self.bce_loss(similarity_scores_flat, labels_flat)
        
        return sc_loss





        
import torch
import torch.nn.functional as F
import clip

# 假设 img_224 是形状为 [16, 3, 224, 224] 的张量
# 假设 cam_224 是形状为 [N * 20, 1, 224, 224] 的张量，其中 N 是批次大小
# 假设 label 是形状为 [16, 20] 的张量

def preprocess(labels):
    new_labels = []
    for n in range(labels.size(0)):
        for idx in range(0, labels.size(1)):
            temp = torch.zeros(1, labels.size(1)).long()
            if labels[n, idx] == 1:
                temp[0, idx] = 1
            new_labels.append(temp)
    return torch.cat(new_labels, dim=0).cuda()

# 初始化所需的组件
device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_fn = clip.load('RN50', device=device)

# 生成样本数据
N = 16  # 批次大小
clip_input_size = 224  # CLIP 模型输入大小
img_224 = torch.randn(N, 3, clip_input_size, clip_input_size).to(device)
cam_224 = torch.randn(N * 20, 1, clip_input_size, clip_input_size).to(device)
label = torch.randint(0, 2, (N, 20)).float().to(device)

# 处理标签
fg_label = preprocess(label.cpu())

# 生成前景和背景图像
fg_224_eval = []
bg_224_eval = []
temp_idx = torch.nonzero(label == 1, as_tuple=False)
for j in range(temp_idx.shape[0]):
    fg_224_eval.append(cam_224[temp_idx[j, 0] * 20 + temp_idx[j, 1]] * img_224[temp_idx[j, 0]])
    bg_224_eval.append((1 - cam_224[temp_idx[j, 0] * 20 + temp_idx[j, 1]]) * img_224[temp_idx[j, 0]])

fg_224_eval = torch.stack(fg_224_eval, dim=0)
bg_224_eval = torch.stack(bg_224_eval, dim=0)

# 对生成的前景图像使用 CLIP 编码
clip_features_fg = clip_model.encode_image(fg_224_eval)
clip_features_bg = clip_model.encode_image(bg_224_eval)

# 输出特征向量的形状
print("Foreground features shape:", clip_features_fg.shape)
print("Background features shape:", clip_features_bg.shape)

# 确保生成的特征向量的形状为 [20, 512]
assert clip_features_fg.shape == (20, 512), "Foreground features shape is incorrect!"
assert clip_features_bg.shape == (20, 512), "Background features shape is incorrect!"


import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# 生成样本数据
N = 16  # 批次大小
clip_input_size = 224  # CLIP 模型输入大小
img_224 = torch.randn(N, 3, clip_input_size, clip_input_size).to(device)
cam_224 = torch.randn(N * 20, 1, clip_input_size, clip_input_size).to(device)
label = torch.randint(0, 2, (N, 20)).float().to(device)

# 处理标签
fg_label = preprocess(label.cpu())

# 生成前景和背景图像
fg_224_eval = []
bg_224_eval = []
temp_idx = torch.nonzero(label == 1, as_tuple=False)
for j in range(temp_idx.shape[0]):
    fg_224_eval.append(cam_224[temp_idx[j, 0] * 20 + temp_idx[j, 1]] * img_224[temp_idx[j, 0]])
    bg_224_eval.append((1 - cam_224[temp_idx[j, 0] * 20 + temp_idx[j, 1]]) * img_224[temp_idx[j, 0]])

fg_224_eval = torch.stack(fg_224_eval, dim=0)
bg_224_eval = torch.stack(bg_224_eval, dim=0)

# 对生成的前景图像使用 CLIP 编码
clip_features_fg = clip_model.encode_image(fg_224_eval)
clip_features_bg = clip_model.encode_image(bg_224_eval)

# 定义对比损失
contrastive_loss = ContrastiveLoss(margin=1.0)

# 标签变换为对比损失所需格式
# 1 表示前景和背景特征不同，0 表示相同
contrastive_labels = torch.ones(clip_features_fg.size(0)).to(device)

# 计算损失
loss = contrastive_loss(clip_features_fg, clip_features_bg, contrastive_labels)

# 输出损失
print("Contrastive Loss:", loss.item())