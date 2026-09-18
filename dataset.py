# dataset.py
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import config

def load_json_dataset(root_json_path: Path):
    train_rows, test_rows = [], []
    json_files = sorted([p for p in root_json_path.iterdir() if p.suffix == ".json"])
    if not json_files:
        raise FileNotFoundError("JSON files missing in the specified folder!")

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            jd = json.load(f)

        prefix = Path.cwd() / "dataset" / jd["meta"]["prefix"]

        for split in ["train", "test"]:
            target_list = train_rows if split == "train" else test_rows
            for rec in jd.get(split, []):
                img_path = prefix / rec["image_path"]
                mask_path = prefix / rec["mask_path"] if rec.get("mask_path") else None
                target_list.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path else None,
                    "category": rec["category"],
                    "anomaly_class": rec["anomaly_class"],
                })

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)

class MultiTypeFcddDataset(Dataset):
    def __init__(self, df, anomaly_class_list, img_size=config.IMG_SIZE, transform=None):
        self.df = df.reset_index(drop=True)
        self.anomaly_class_list = anomaly_class_list
        self.class_to_index = {c: i for i, c in enumerate(anomaly_class_list)}
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = row["anomaly_class"]

        if self.transform:
            img = self.transform(img)

        target = torch.zeros(len(self.anomaly_class_list))
        if label in self.class_to_index:
            target[self.class_to_index[label]] = 1.0

        return img, target, label

class BalancedClassSampler(Sampler):
    def __init__(self, df, classes, shuffle=True):
        self.df = df
        self.classes = classes
        self.shuffle = shuffle

        self.class_to_indices = {cls: df.index[df["anomaly_class"] == cls].tolist() for cls in classes}
        if shuffle:
            for cls in classes:
                random.shuffle(self.class_to_indices[cls])

        self.position = {cls: 0 for cls in classes}
        self.total_len = len(df)

    def __len__(self):
        return self.total_len

    def __iter__(self):
        seen = 0
        seen_flags = {cls: np.zeros(len(self.class_to_indices[cls]), dtype=bool) for cls in self.classes}

        while seen < self.total_len:
            cls = random.choice(self.classes)
            if self.position[cls] >= len(self.class_to_indices[cls]):
                if self.shuffle: 
                    random.shuffle(self.class_to_indices[cls])
                self.position[cls] = 0

            local_pos = self.position[cls]
            global_idx = self.class_to_indices[cls][local_pos]

            if not seen_flags[cls][local_pos]:
                seen_flags[cls][local_pos] = True
                seen += 1

            self.position[cls] += 1
            yield global_idx

train_transform = T.Compose([
    T.RandomApply([
        T.RandomRotation(15),
        T.RandomAffine(0, translate=(20/config.IMG_SIZE[0], 20/config.IMG_SIZE[1])),
        T.ColorJitter(brightness=0.3, contrast=0.3),
    ], p=0.5),
    T.Resize(config.IMG_SIZE),
    T.ToTensor(),
])

test_transform = T.Compose([
    T.Resize(config.IMG_SIZE),
    T.ToTensor(),
])