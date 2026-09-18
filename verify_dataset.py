# verify_dataset.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset import load_json_dataset, MultiTypeFcddDataset, BalancedClassSampler, train_transform

def print_dataset_summary(df_train, df_test, anomaly_classes):
    train_counts = df_train["anomaly_class"].value_counts()
    test_counts = df_test["anomaly_class"].value_counts()

    stats = []
    for cls in anomaly_classes + ["OK"]:
        stats.append({
            "Class Name": cls,
            "Train Images": train_counts.get(cls, 0),
            "Test Images": test_counts.get(cls, 0),
            "Total Images": train_counts.get(cls, 0) + test_counts.get(cls, 0)
        })

    df_stats = pd.DataFrame(stats).set_index("Class Name")
    df_stats.loc["TOTAL"] = df_stats.sum()

    print("\n" + "="*55)
    print("        REAL-IAD DATASET CLASS DISTRIBUTION")
    print("="*55)
    print(df_stats.to_string())
    print("="*55 + "\n")

def show_sample_batch(loader, title="Sample batch", num_images=8):
    imgs, targets, labels = next(iter(loader))
    fig, axes = plt.subplots(1, num_images, figsize=(3*num_images, 3))
    fig.suptitle(title, fontsize=16)

    for i in range(num_images):
        ax = axes[i]
        img_np = imgs[i].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(str(labels[i]))

    plt.tight_layout()
    plt.savefig("sample_verification_batch.png")
    print("Sample visualization batch exported successfully as 'sample_verification_batch.png'")

def main():
    print("Loading data metadata frames and generating dataset distribution matrix table...")
    df_train, df_test = load_json_dataset(config.ROOT_JSON_PATH)
    df_train = df_train[df_train["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)
    df_test = df_test[df_test["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)

    all_classes = sorted(df_train["anomaly_class"].unique())
    classes_ordered = [c for c in all_classes if c != "OK"] + ["OK"] if "OK" in all_classes else all_classes
    anomaly_classes = [c for c in classes_ordered if c != "OK"]

    print_dataset_summary(df_train, df_test, anomaly_classes)

    train_ds = MultiTypeFcddDataset(df_train, anomaly_classes, config.IMG_SIZE, train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=BalancedClassSampler(df_train, classes_ordered), num_workers=0
    )

    print("Drawing sample dataset visualization grid...")
    show_sample_batch(train_loader, "Training samples verification", 8)

if __name__ == "__main__":
    main()