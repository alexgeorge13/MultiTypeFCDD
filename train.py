# train.py
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import config
from dataset import load_json_dataset, MultiTypeFcddDataset, BalancedClassSampler, train_transform, test_transform
from model import FCDDNet
from loss import multi_type_fcdd_loss

def set_seed(seed=42):
    """Enforces absolute numerical repeatability across CPU, GPU, and library configurations"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def run_training():
    print("Training a NEW model...")
    print("Starting training...")

    df_train, df_test = load_json_dataset(config.ROOT_JSON_PATH)
    df_train = df_train[df_train["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)

    all_classes = sorted(df_train["anomaly_class"].unique())
    classes_ordered = [c for c in all_classes if c != "OK"] + ["OK"] if "OK" in all_classes else all_classes
    anomaly_classes = [c for c in classes_ordered if c != "OK"]
    num_anomaly = len(anomaly_classes)

    train_ds = MultiTypeFcddDataset(df_train, anomaly_classes, config.IMG_SIZE, train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=BalancedClassSampler(df_train, classes_ordered), num_workers=0
    )

    df_test = df_test[df_test["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)
    val_ds = MultiTypeFcddDataset(df_test, anomaly_classes, config.IMG_SIZE, test_transform)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    fcdd_model = FCDDNet(num_anomaly=num_anomaly).to(config.DEVICE)
    optimizer = optim.Adam(fcdd_model.parameters(), lr=config.LEARNING_RATE)

    for current_epoch in range(config.NUM_EPOCHS):
        fcdd_model.train()
        running_train_loss = 0
        start_time = time.time()

        for iteration, (imgs, tvec, _) in enumerate(train_loader, start=1):
            imgs, tvec = imgs.to(config.DEVICE), tvec.to(config.DEVICE)

            out = fcdd_model(imgs)
            B, C, H, W = out.shape
            tmap = tvec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

            loss = multi_type_fcdd_loss(out, tmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            elapsed = time.time() - start_time
            it_s = iteration / elapsed if elapsed > 0 else 0

            print(
                f"\rEpoch {current_epoch+1}/{config.NUM_EPOCHS}  |  "
                f"Iteration {iteration} ({it_s:.2f} it/s) |  "
                f"Time: {format_time(elapsed)}  |  "
                f"Train Loss: {running_train_loss/iteration:.4f}",
                end=""
            )
        print()

        fcdd_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs, val_tvec, _ in val_loader:
                val_imgs, val_tvec = val_imgs.to(config.DEVICE), val_tvec.to(config.DEVICE)
                val_out = fcdd_model(val_imgs)
                _, _, val_H, val_W = val_out.shape
                val_tmap = val_tvec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, val_H, val_W)
                val_loss += multi_type_fcdd_loss(val_out, val_tmap).item()
        val_loss /= len(val_loader)
        print(f"\n[Validation] Epoch {current_epoch+1}/{config.NUM_EPOCHS} | Iteration {iteration} | Val Loss: {val_loss:.4f}")
        fcdd_model.train()

    print("Training finished.")
    print("Model and weights artifacts saved.")
    torch.save({
        'model_state_dict': fcdd_model.state_dict(),
        'anomaly_classes': anomaly_classes,
    }, config.SAVE_NAME)

if __name__ == "__main__":
    # Freeze environment randomness before execution setup begins
    set_seed(config.SEED)
    run_training()