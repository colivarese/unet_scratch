import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import CarvanaDataset

if __name__ == "__main__":
    LR = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 2
    DATA_PATH = "unet"
    MODEL_SAVE_PATH = "unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CarvanaDataset(DATA_PATH)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_dim=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            img = batch[0].float().to(device)
            mask = batch[1].float().to(device)

            y_pred = model(img)

            optimizer.zero_grad()
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        train_loss = train_running_loss / (idx+1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                img = batch[0].float().to(device)
                mask = batch[1].flaot().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx+1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)