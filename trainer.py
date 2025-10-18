import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


CSV_PATH = './data/Dataset/EPIC_100_train.csv'
FRAME_ROOT = './data/Dataset/frames'
SAVE_MODEL_PATH = '/content/KitchenActionPrediction2/lstm_action_model.pth'
LOSS_LOG_PATH = '/content/KitchenActionPrediction2/training_loss.csv'

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3   
SEQUENCE_LENGTH = 16
IMG_SIZE = 128



class EpicKitchensDataset(Dataset):
    def __init__(self, csv_path, frames_root, transform=None, num_frames=SEQUENCE_LENGTH):
        self.data = pd.read_csv(csv_path)
        self.frames_root = frames_root
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        verb_class = int(row['verb_class'])
        noun_class = int(row['noun_class'])

        video_folder = os.path.join(self.frames_root, video_id)
        frame_indices = torch.linspace(start_frame, stop_frame, self.num_frames, dtype=torch.int)

        frames = []
        for f in frame_indices:
            frame_path = os.path.join(video_folder, f'frame_{f:010d}.jpg')
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)

        if len(frames) == 0:
            frames = [torch.zeros(3, IMG_SIZE, IMG_SIZE) for _ in range(self.num_frames)]

        return frames, torch.tensor(verb_class), torch.tensor(noun_class)


def pad_collate(batch):
    frames_batch, verb_batch, noun_batch = zip(*batch)
    max_len = max([len(frames) for frames in frames_batch])
    padded_frames = []

    for frames in frames_batch:
        pad_len = max_len - len(frames)
        if pad_len > 0:
            pad_frames = [torch.zeros_like(frames[0]) for _ in range(pad_len)]
            frames = frames + pad_frames
        padded_frames.append(torch.stack(frames))

    padded_frames = torch.stack(padded_frames)
    verb_batch = torch.stack(verb_batch)
    noun_batch = torch.stack(noun_batch)
    return padded_frames, verb_batch, noun_batch



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          stride=1, padding=(kernel_size - 1) * dilation_size,
                          dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNActionModel(nn.Module):
    def __init__(self, hidden_size=256, num_verb_classes=97, num_noun_classes=300):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.tcn = TemporalConvNet(num_inputs=64, num_channels=[128, hidden_size])
        self.verb_head = nn.Linear(hidden_size, num_verb_classes)
        self.noun_head = nn.Linear(hidden_size, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = []
        for t in range(T):
            f = self.cnn(x[:, t])
            features.append(f.squeeze(-1).squeeze(-1))
        features = torch.stack(features, dim=2)
        tcn_out = self.tcn(features)
        last_out = tcn_out[:, :, -1]
        verb_logits = self.verb_head(last_out)
        noun_logits = self.noun_head(last_out)
        return verb_logits, noun_logits



def top1_accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def top5_accuracy(preds, labels):
    top5 = preds.topk(5, dim=1).indices
    correct = top5.eq(labels.view(-1, 1)).sum().item()
    return correct / labels.size(0)

def action_pair_accuracy(verb_preds, noun_preds, verb_labels, noun_labels):
    verb_correct = verb_preds.argmax(dim=1) == verb_labels
    noun_correct = noun_preds.argmax(dim=1) == noun_labels
    both_correct = (verb_correct & noun_correct).float().mean().item()
    return both_correct



def train_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = EpicKitchensDataset(CSV_PATH, FRAME_ROOT, transform)
    val_split = 0.1
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, collate_fn=pad_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNActionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    loss_log = []

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        for frames, verb_labels, noun_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)

            optimizer.zero_grad()
            verb_preds, noun_preds = model(frames)
            loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        
        model.eval()
        total_val_loss = 0.0
        total_action_acc, total_verb_top1, total_noun_top1 = 0, 0, 0
        total_verb_top5, total_noun_top5 = 0, 0
        num_batches = 0

        with torch.no_grad():
            for frames, verb_labels, noun_labels in val_loader:
                frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)
                verb_preds, noun_preds = model(frames)
                loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
                total_val_loss += loss.item()

                total_action_acc += action_pair_accuracy(verb_preds, noun_preds, verb_labels, noun_labels)
                total_verb_top1 += top1_accuracy(verb_preds, verb_labels)
                total_noun_top1 += top1_accuracy(noun_preds, noun_labels)
                total_verb_top5 += top5_accuracy(verb_preds, verb_labels)
                total_noun_top5 += top5_accuracy(noun_preds, noun_labels)
                num_batches += 1

        avg_val_loss = total_val_loss / len(val_loader)
        avg_action_acc = total_action_acc / num_batches
        avg_verb_top1 = total_verb_top1 / num_batches
        avg_noun_top1 = total_noun_top1 / num_batches
        avg_verb_top5 = total_verb_top5 / num_batches
        avg_noun_top5 = total_noun_top5 / num_batches

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] "
              f"- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Verb Top1: {avg_verb_top1:.4f} | Noun Top1: {avg_noun_top1:.4f}")
        print(f"Verb Top5: {avg_verb_top5:.4f} | Noun Top5: {avg_noun_top5:.4f}")
        print(f"Action Pair Accuracy: {avg_action_acc:.4f}")

        # ---- Save Best Model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"New best model saved (Val Loss: {avg_val_loss:.4f})")

        loss_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'action_acc': avg_action_acc
        })

    pd.DataFrame(loss_log).to_csv(LOSS_LOG_PATH, index=False)
    print(f"\nTraining complete. Best model saved to {SAVE_MODEL_PATH}")


if __name__ == '__main__':
    train_model()
