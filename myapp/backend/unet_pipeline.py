import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

INPUT_DIR  = "input"
TARGET_DIR = "target"
SAVE_PATH  = "backend/runs/unet/ecg_unet.pth"

IMG_SIZE   = 512
BATCH_SIZE = 2
EPOCHS     = 300
LR          = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# STEP 1: TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()   # [0,1]
])

# ======================================================
# STEP 2: DATASET
# ======================================================
class ECGDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.files = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        inp = cv2.imread(os.path.join(self.input_dir, name), cv2.IMREAD_GRAYSCALE)
        tgt = cv2.imread(os.path.join(self.target_dir, name), cv2.IMREAD_GRAYSCALE)

        if inp is None or tgt is None:
            raise ValueError(f"Error reading {name}")

        if self.transform:
            inp = self.transform(inp)
            tgt = self.transform(tgt)

        return inp, tgt

dataset = ECGDataset(INPUT_DIR, TARGET_DIR, transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))

# ======================================================
# STEP 3: U-NET (FIXED OUTPUT)
# ======================================================
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ECGUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.mid = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m  = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

model = ECGUNet().to(device)
print("Model initialized")

# ======================================================
# STEP 4: TRAINING SETUP
# ======================================================
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================================================
# STEP 5: TRAINING
# ======================================================
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("Model saved at:", SAVE_PATH)

# ======================================================
# STEP 6: TEST ONE IMAGE
# ======================================================
model.eval()
with torch.no_grad():
    inp, tgt = dataset[0]
    out = model(inp.unsqueeze(0).to(device))

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(inp.squeeze(), cmap="gray")
plt.title("INPUT")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(out.squeeze().cpu(), cmap="gray")
plt.title("OUTPUT (U-Net)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(tgt.squeeze(), cmap="gray")
plt.title("TARGET")
plt.axis("off")

plt.show()
