# evaluate.py
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.measure import label as label_regions

import config
from dataset import load_json_dataset, MultiTypeFcddDataset, test_transform
from model import FCDDNet


def compute_aupro(gt_masks, anomaly_maps, max_fpr=0.3, num_steps=1000):
    """
    Computes Area Under Per-Region Overlap (AUPRO) integrated up to max_fpr (default 0.3).
    - gt_masks: np.ndarray of shape (N, H, W) binary ground truth masks
    - anomaly_maps: np.ndarray of shape (N, H, W) continuous anomaly maps
    """
    labeled_regions = []
    num_regions = 0
    for mask in gt_masks:
        if mask.sum() > 0:
            labeled, num_feat = label_regions(mask > 0, return_num=True)
            labeled_regions.append(labeled)
            num_regions += num_feat
        else:
            labeled_regions.append(np.zeros_like(mask, dtype=int))

    if num_regions == 0:
        return 0.0

    region_scores = []
    for mask_labeled, map_arr in zip(labeled_regions, anomaly_maps):
        for reg_id in range(1, mask_labeled.max() + 1):
            region_scores.append(map_arr[mask_labeled == reg_id])

    normal_scores = anomaly_maps[gt_masks == 0]
    min_val, max_val = anomaly_maps.min(), anomaly_maps.max()
    thresholds = np.linspace(min_val, max_val, num_steps)

    pros = []
    fprs = []

    for th in reversed(thresholds):
        fpr = np.mean(normal_scores >= th) if len(normal_scores) > 0 else 0.0
        fprs.append(fpr)

        region_overlaps = [np.mean(scores >= th) for scores in region_scores]
        pros.append(np.mean(region_overlaps))

        if fpr >= max_fpr:
            break

    fprs = np.array(fprs)
    pros = np.array(pros)

    if fprs.max() > 0:
        norm_fprs = fprs / max_fpr
        aupro = np.trapz(pros, norm_fprs)
        return float(np.clip(aupro, 0.0, 1.0))
    return 0.0


def evaluate_and_compute_metrics():
    print("Evaluating MultiTypeFCDD model against test split...")
    _, df_test = load_json_dataset(config.ROOT_JSON_PATH)
    df_test = df_test[df_test["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)

    print(f"Found {len(df_test)} test samples matching categories: {config.SELECTED_OBJECTS}")
    if len(df_test) == 0:
        raise ValueError("No matching test samples found! Check config.SELECTED_OBJECTS vs JSON category names.")

    checkpoint = torch.load(config.SAVE_NAME, map_location=config.DEVICE, weights_only=False)
    anomaly_classes = checkpoint['anomaly_classes']
    num_anomaly = len(anomaly_classes)

    test_ds = MultiTypeFcddDataset(df_test, anomaly_classes, config.IMG_SIZE, test_transform)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    fcdd_model = FCDDNet(num_anomaly=num_anomaly).to(config.DEVICE)
    fcdd_model.load_state_dict(checkpoint['model_state_dict'])
    fcdd_model.eval()

    min_map_vals = np.full((num_anomaly,), np.inf)
    max_map_vals = np.full((num_anomaly,), -np.inf)

    all_img_targets = []
    all_img_scores = []
    all_gt_masks = []
    all_pred_maps = []

    img_idx = 0
    with torch.no_grad():
        for imgs, tvec, _ in test_loader:
            imgs = imgs.to(config.DEVICE)
            outputs = fcdd_model(imgs)
            outputs_np = outputs.cpu().numpy()

            outputs_upsampled = F.interpolate(
                outputs, size=config.IMG_SIZE, mode="bilinear", align_corners=False
            ).cpu().numpy()

            for c in range(num_anomaly):
                class_maps = outputs_np[:, c, :, :]
                min_map_vals[c] = min(min_map_vals[c], class_maps.min())
                max_map_vals[c] = max(max_map_vals[c], class_maps.max())

            img_scores = outputs_np.max(axis=(2, 3))
            all_img_scores.append(img_scores)
            all_img_targets.append(tvec.numpy())

            for b in range(imgs.size(0)):
                row = df_test.iloc[img_idx]
                mask_p = row.get("mask_path", None)
                if isinstance(mask_p, str) and mask_p.strip() and Path(mask_p).exists():
                    gt_mask = np.array(Image.open(mask_p).convert("L").resize(config.IMG_SIZE)) > 0
                else:
                    gt_mask = np.zeros(config.IMG_SIZE, dtype=bool)

                all_gt_masks.append(gt_mask)
                all_pred_maps.append(outputs_upsampled[b])
                img_idx += 1

    all_img_scores = np.concatenate(all_img_scores, axis=0)
    all_img_targets = np.concatenate(all_img_targets, axis=0)
    all_gt_masks = np.stack(all_gt_masks, axis=0)
    all_pred_maps = np.stack(all_pred_maps, axis=0)

    metrics_per_class = []

    print("\n" + "=" * 65)
    print(" MULTITYPE-FCDD EVALUATION METRICS (Real-IAD Dataset)")
    print("=" * 65)

    for c, cls_name in enumerate(anomaly_classes):
        # 1. Image-level AUROC (I-AUROC)
        y_true_img = all_img_targets[:, c]
        y_score_img = all_img_scores[:, c]
        i_auroc = roc_auc_score(y_true_img, y_score_img) if len(np.unique(y_true_img)) > 1 else np.nan

        # 2. Pixel-level AUROC (P-AUROC)
        class_map = all_pred_maps[:, c, :, :]
        p_auroc = roc_auc_score(all_gt_masks.ravel(), class_map.ravel())

        # 3. AUPRO
        aupro = compute_aupro(all_gt_masks, class_map, max_fpr=0.3)

        metrics_per_class.append({
            "Anomaly Class": cls_name,
            "I-AUROC": i_auroc,
            "P-AUROC": p_auroc,
            "AUPRO": aupro
        })

    df_metrics = pd.DataFrame(metrics_per_class)
    
    mean_row = {
        "Anomaly Class": "Mean",
        "I-AUROC": df_metrics["I-AUROC"].mean(),
        "P-AUROC": df_metrics["P-AUROC"].mean(),
        "AUPRO": df_metrics["AUPRO"].mean()
    }
    df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)

    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x*100:.2f}%"))
    print("=" * 65)

    torch.save({
        'model_state_dict': fcdd_model.state_dict(),
        'min_map_vals': min_map_vals,
        'max_map_vals': max_map_vals,
        'anomaly_classes': anomaly_classes,
        'eval_metrics': df_metrics.to_dict()
    }, config.SAVE_NAME)
    print(f"\nCheckpoint successfully updated with min/max bounds and metrics at '{config.SAVE_NAME}'.")


if __name__ == "__main__":
    evaluate_and_compute_metrics()