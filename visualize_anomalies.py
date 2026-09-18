# visualize_anomalies.py
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import config
from dataset import load_json_dataset, MultiTypeFcddDataset, test_transform
from model import FCDDNet

# Defect-specific base colors (same order as your anomaly_classes)
BASE_COLORS = np.array([
    [0.0, 1.0, 1.0],    # AK
    [0.18, 0.55, 0.34], # BX
    [1.0,  1.0, 0.1],   # CH
    [1.0,  0.1, 0.6],   # HS
    [0.15, 0.6, 0.85],  # PS
    [1.0,  0.5, 0.0],   # QS
    [0.9,  0.0, 0.1],   # YW
    [0.5,  0.3, 1.0],   # ZW
], dtype=np.float32)

NUM_STEPS = 256
# Build class-specific colormaps: black → class_color
CLASS_CMAPS = np.stack([
    np.column_stack([
        np.linspace(0, color[0], NUM_STEPS),
        np.linspace(0, color[1], NUM_STEPS),
        np.linspace(0, color[2], NUM_STEPS),
    ])
    for color in BASE_COLORS
])

def visualise_fcdd_heatmaps_grid_colored(model, img_tensor, anomaly_classes,
                                         min_vals, max_vals, mask_path=None,
                                         gt_label=None, object_name=None, device="cpu", upsample="bilinear",
                                         blend_factor=2.0, output_name="heatmap_output.png"):
    font_size = 20
    
    model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        maps = model(x).squeeze(0)               # (C, H, W)

    # Input image
    img_np = (img_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

    # Main plot grid configuration - modified to include the object category name dynamically
    fig, axes = plt.subplots(2,5, figsize=(20,8))
    axes = axes.flatten()
    
    supt_title = f"MultiTypeFCDD Predictions"
    if object_name:
        supt_title += f" | Object: {object_name}"
    fig.suptitle(supt_title, fontsize=24, y=0.98)

    # Display input image
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Input", fontsize=font_size)

    # Colorised ground truth mask 
    gt_title = f"GT Mask: {gt_label}" if gt_label is not None else "GT Mask"
    gt_overlay = np.zeros_like(img_np, dtype=np.float32)
    
    if mask_path and Path(mask_path).exists():
        gt_mask = np.array(
            Image.open(mask_path)
            .convert("L")
            .resize(img_np.shape[:2][::-1])
        )
        gt_binary = (gt_mask > 0).astype(np.float32)
    
        if gt_label in anomaly_classes:
            cls_idx = anomaly_classes.index(gt_label)
            gt_color = BASE_COLORS[cls_idx]
        else:
            gt_color = np.array([1.0, 1.0, 1.0])
    
        gt_overlay = gt_binary[..., None] * gt_color[None, None, :]
    
    axes[5].imshow(gt_overlay)
    axes[5].axis("off")
    axes[5].set_title(gt_title, fontsize=font_size)

    # Generate colored anomaly heatmaps
    for i, cls_name in enumerate(anomaly_classes):
        cmap = CLASS_CMAPS[i]

        # 1) Upsample map from model
        up = F.interpolate(maps[i][None,None], size=img_np.shape[:2],
                           mode=upsample, align_corners=False)\
             .squeeze().cpu().numpy()

        # 2) Normalise using class-specific min/max
        map_norm = np.clip((up - min_vals[i]) /
                           (max_vals[i] - min_vals[i] + 1e-8), 0, 1)

        # 3) Build RGB heatmap by mapping normalised map to index 0–255
        idx = (map_norm * 255).astype(np.uint8)
        heatmap_rgb = cmap[idx]

        # 4) Overlay: input image + heatmap * blend_factor
        img_float = img_np.astype(np.float32) / 255.0
        overlay = img_float * 0.4 + heatmap_rgb * (map_norm[...,None] * blend_factor)
        overlay = np.clip(overlay, 0, 1)

        # 5) Correct subplot index
        subplot_idx = i+1 if i < 4 else i+2

        axes[subplot_idx].imshow(overlay)
        axes[subplot_idx].axis("off")
        axes[subplot_idx].set_title(cls_name, fontsize=font_size)

    # Fill empty tiles (if fewer than 8 classes)
    for j in range(len(anomaly_classes)+1, 10):
        if len(axes[j].images) == 0:
            axes[j].imshow(np.zeros_like(img_np))
            axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"Heatmap grid exported successfully to '{output_name}'")

def main():
    sample_idx = config.VISUALIZATION_SAMPLE_IDX
    
    print(f"Loading visualization mapping for sample index from config: {sample_idx}...")
    _, df_test = load_json_dataset(config.ROOT_JSON_PATH)
    df_test = df_test[df_test["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)

    if sample_idx >= len(df_test):
        raise IndexError(f"Requested index {sample_idx} is out of bounds for test set size ({len(df_test)}).")

    if not Path(config.SAVE_NAME).exists():
        raise FileNotFoundError(f"Weights file missing: {config.SAVE_NAME}. Run evaluate.py first.")
        
    checkpoint = torch.load(config.SAVE_NAME, map_location=config.DEVICE, weights_only=False)
    anomaly_classes = checkpoint['anomaly_classes']
    min_map_vals = checkpoint['min_map_vals']
    max_map_vals = checkpoint['max_map_vals']

    fcdd_model = FCDDNet(num_anomaly=len(anomaly_classes)).to(config.DEVICE)
    fcdd_model.load_state_dict(checkpoint['model_state_dict'])

    test_ds = MultiTypeFcddDataset(df_test, anomaly_classes, config.IMG_SIZE, test_transform)
    img_tensor, _, label_str = test_ds[sample_idx]
    
    # Extract object category name metadata
    object_name = df_test.iloc[sample_idx]["category"]
    mask_path = df_test.iloc[sample_idx]["mask_path"]
    
    # If the value is missing or parsed as a float NaN, convert it safely to None
    if isinstance(mask_path, float) or not mask_path:
        mask_path = None

    # Save output using index name combined with the category name string
    output_filename = f"visualized_sample_{sample_idx}_{object_name}.png"

    visualise_fcdd_heatmaps_grid_colored(
        model=fcdd_model, 
        img_tensor=img_tensor, 
        anomaly_classes=anomaly_classes, 
        min_vals=min_map_vals, 
        max_vals=max_map_vals,
        mask_path=mask_path, 
        gt_label=label_str, 
        object_name=object_name,
        device=config.DEVICE, 
        blend_factor=10,
        output_name=output_filename
    )

if __name__ == "__main__":
    main()