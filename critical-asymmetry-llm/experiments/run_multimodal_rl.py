import torch
import yaml


def train(config_path="configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print("Configuration chargee.")
    print("Entrainement multi-modal pret.")
    return None


if __name__ == "__main__":
    train()
