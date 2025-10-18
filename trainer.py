import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


CSV_PATH = 'EPIC_100_train.csv'
FRAME_ROOT = 'frames'
SAVE_MODEL_PATH = 'tcn_action_model.pth'
LOSS_LOG_PATH = 'training_loss_tcn.csv'

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 16
IMG_SIZE = 128


class EPICKitchensDataset(Dataset):
    def __init__(self, csv_path, frame_root, transform=None):
        self.data = pd.read_csv(csv_path)
        self.frame_root = frame_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        verb_class = int(row['verb_class'])
        noun_class = int(row['noun_class'])

        frame_dir = os.path.join(self.frame_root, video_id)
        frames = []
        frame_ids = range(start_frame, min(stop_frame, start_frame + SEQUENCE_LENGTH))
        for i in frame_ids:
            frame_path = os.path.join(frame_dir, f'frame_{i:010d}.jpg')
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)

        if len(frames) < SEQUENCE_LENGTH:
            frames += [frames[-1]] * (SEQUENCE_LENGTH - len(frames))

        frames = torch.stack(frames)
        return frames, torch.tensor(verb_class), torch.tensor(noun_class)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                 stride=1, padding=(kernel_size-1)*dilation_size,
                                 dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNActionModel(nn.Module):
    def __init__(self, hidden_size=256, num_verb_classes=97, num_noun_classes=300):
        super().__init__()
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # TCN for temporal modeling
        self.tcn = TemporalConvNet(num_inputs=64, num_channels=[128, hidden_size])
        # Classification heads
        self.verb_head = nn.Linear(hidden_size, num_verb_classes)
        self.noun_head = nn.Linear(hidden_size, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = []
        for t in range(T):
            f = self.cnn(x[:, t])
            features.append(f.squeeze(-1).squeeze(-1))
        features = torch.stack(features, dim=2)  # (B, 64, T)

        tcn_out = self.tcn(features)  # (B, hidden_size, T)
        last_out = tcn_out[:, :, -1]  # use last timestep

        verb_logits = self.verb_head(last_out)
        noun_logits = self.noun_head(last_out)
        return verb_logits, noun_logits

def train_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Split dataset
    full_dataset = EPICKitchensDataset(CSV_PATH, FRAME_ROOT, transform)
    val_split = 0.1
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNActionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    loss_log = []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_train_loss = 0.0
        for frames, verb_labels, noun_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)

            optimizer.zero_grad()
            verb_preds, noun_preds = model(frames)
            loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for frames, verb_labels, noun_labels in val_loader:
                frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)
                verb_preds, noun_preds = model(frames)
                loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"New best TCN model saved (Val Loss: {avg_val_loss:.4f})")

        loss_log.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

    pd.DataFrame(loss_log).to_csv(LOSS_LOG_PATH, index=False)
    print(f"Training complete. Best TCN model saved to {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()
