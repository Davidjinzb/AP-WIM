# %%
from datasets import AudioMNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import random
from tqdm.auto import tqdm
from torch import optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(42)

# %%
def build_resnet18_audiomnist():
    model = models.resnet18(pretrained=False)

    # 输入：1 通道 log-mel
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # 输出：50 类
    model.fc = nn.Linear(model.fc.in_features, 60)

    return model
model = build_resnet18_audiomnist()
model = model.to(device)

# %%
train_dataset = AudioMNIST(mode='train')
test_dataset = AudioMNIST(mode='test')

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8
)

# %%
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0   # Vanilla：不加正则
)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y, *_ in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        # print(x.shape)
        # raise Exception

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y, *_ in tqdm(loader, desc="Testing", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


# %%
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion
    )

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    # # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "resnet18_audiomnist_vanilla.pth")
        print("✅ Saved best model")


# %%


# %%


# %%



