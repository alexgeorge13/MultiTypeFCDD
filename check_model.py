# check_model.py
import os
import torch
from torchinfo import summary

# Suppress unauthenticated HF hub alerts
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import config
from dataset import load_json_dataset
from model import FCDDNet

def main():
    print("Parsing dataset labels to evaluate network anomaly classes...")
    df_train, _ = load_json_dataset(config.ROOT_JSON_PATH)
    df_train = df_train[df_train["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)
    
    all_classes = sorted(df_train["anomaly_class"].unique())
    anomaly_classes = [c for c in all_classes if c != "OK"]
    num_anomaly = len(anomaly_classes)

    print(f"Instantiating model on target backend device: {config.DEVICE}")
    print(f"Number of anomaly channels: {num_anomaly}\n")
    
    fcdd_model = FCDDNet(num_anomaly=num_anomaly).to(config.DEVICE)
    
    # Render full summary layout cleanly
    summary(fcdd_model, input_size=(1, 3, config.IMG_SIZE[0], config.IMG_SIZE[1]))

if __name__ == "__main__":
    main()