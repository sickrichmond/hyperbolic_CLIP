import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data.iab_clip_dataset import IABCLIPDataset
from models.det_on_frozen_CLIP import DetectorDF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_path", required=True)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train, y_train, classes_train = torch.load(Path(args.feature_path) / "clip_features_train.pt")
    X_val, y_val, classes_val = torch.load(Path(args.feature_path) / "clip_features_val.pt")

    model = nn.Linear(X_train.shape[1], len(classes_train))
    
    



if __name__ == "__main__": 
    main()