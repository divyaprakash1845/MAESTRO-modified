import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from dataset import MultiSubjectNeuroFlowDataset
from model import MAESTRO

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 INITIALIZING MAESTRO PIPELINE ON: {device}")

    # Looks one directory up for the raw_data folder
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    
    suds_scores = [25, 25, 35, 30, 25]
    boundaries = [0, 690, 1376, 2500, 3668, 4306] 
    
    print("🔄 Loading and Slicing Dataset...")
    full_ds = MultiSubjectNeuroFlowDataset(raw_data_dir, suds_scores, boundaries)
    
    # ... rest of your training code remains exactly the same ...
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    
    model = MAESTRO().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 
    
    epochs = 15
    print(f"\n" + "="*40 + f"\n🧠 TRAINING ON {len(full_ds)} DATA WINDOWS\n" + "="*40)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{epochs} | Average MSE Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/maestro_final.pth")
    print("\n💾 Training complete. Weights saved to weights/maestro_final.pth")

if __name__ == "__main__":
    main()
