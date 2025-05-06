import yaml
import random
import torch
import argparse

from trainer import Trainer
from utils import *
from data_process import *

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    """
    Parse Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["GCN", "GAT", "HGT"], required=True, help="Model to train")
    args = parser.parse_args()

    """
    Prepare Graph
    """
    ehr_graph = EHRGraph()
    ehr_graph.process_graph()

    """
    Train Model
    """
    config_name = "./models.yml"

    with open(config_name, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {config_name}")

    trainer = Trainer(config, args.model)
    trainer.train()

        