import torch
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get the mterics from the checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    print(args.checkpoint)
    cap_model_cpt = torch.load(args.checkpoint, map_location='cpu')
    for k in cap_model_cpt['val_1_metrics']['Average across tIoUs'].keys():
        print(k, np.round(100*(cap_model_cpt['val_1_metrics']['Average across tIoUs'][k] + cap_model_cpt['val_2_metrics']['Average across tIoUs'][k])/2.0 , 2))
