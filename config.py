# config.py
from pathlib import Path
import torch

# Random seed for reproducible results
SEED = 42

# Noise ratio for training data (0.1, 0.2, 0.4)
ALPHA = 0.1

# JSON folder selection
JSON_FOLDER = f"realiad_jsons_fuiad_{ALPHA}"
ROOT_JSON_PATH = Path.cwd() / "realiad_jsons" / JSON_FOLDER

# Objects in the Real-IAD dataset - include all 30 objects
SELECTED_OBJECTS = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood",
    "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug",
    "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape",
    "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1", "u_block",
    "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper"
]

# Input size of Inception-ResNet-v2 network
IMG_SIZE = (299, 299)

# Name for the saved model
SAVE_NAME = "trained_multitypeFCDD_net.pth"

# Training parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- VISUALIZATION SETUP ---
# Set the specific test dataset index you want to visualize here
VISUALIZATION_SAMPLE_IDX = 926