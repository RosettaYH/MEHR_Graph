import os
from collections import OrderedDict
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torch import optim

from utils import *

import warnings
warnings.filterwarnings("ignore")
import torch, numpy as np, random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Trainer():
    def __init__(self, config: OrderedDict, model: str):
        self.name = model
        self.config = config
        self.config_data = config["PATHS"]
        self.config_train = config["TRAIN"]
        self.config_optim = config["OPTIMIZERS"][model]
        self.config_gnn = config["MODEL"][model]

        self.n_epoch = self.config_train['num_epochs']
        self.batch_size = self.config_train['batch_size']

        self.gpu_ids = config.get('gpu_ids', '0')
        if self.name == "GAT":
            self.device = "cpu"
        elif torch.cuda.is_available() and self.gpu_ids:
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.use_gpu = self.device != "cpu"
        self.tasks = [ 
            "readm",
            "mort_pred",
            "drug_rec"
        ]

        graph_path = self.config_data["graph_path"]
        labels_path = self.config_data["labels_path"]
        self.graph, self.labels, self.train_mask, self.test_mask = load_graph(graph_path, labels_path)

        self.graph = add_reverse_edges(self.graph)
        self.node_dict = {nt: torch.arange(self.graph[nt].num_nodes) for nt in self.graph.node_types}

        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=self.config_optim['lr'], weight_decay=self.config_optim["weight_decay"])
        
    def train(self) -> None:
        print(f"Start training {self.name}")
        training_range = tqdm(range(self.n_epoch), nrows=3)
        best_test_acc = 0.0
        task_for_monitoring = "drug_rec"

        for epoch in training_range:
            self.gnn.train()
            epoch_stats = {"Epoch": epoch + 1}
            total_loss = 0.0

            self.optimizer.zero_grad()

            for t in self.tasks:
                d = self.node_dict.copy()
                indices = self.train_mask[t]
                d["visit"] = self.node_dict["visit"][indices]
                sg = hetero_subgraph(self.graph, d).to(self.device)

                preds = self.gnn(sg, "visit", t)

                if t == "drug_rec":
                    all_drugs = self.train_mask["all_drugs"]
                    labels = []
                    for i in indices:
                        drugs = self.labels[t][i]
                        labels.append([1 if d in drugs else 0 for d in all_drugs])
                    labels = torch.FloatTensor(labels).to(self.device)
                    loss = F.binary_cross_entropy_with_logits(preds, labels)
                else:
                    labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)
                    loss = F.cross_entropy(preds, labels)

                total_loss += loss

            total_loss.backward()
            self.optimizer.step()

            d = self.node_dict.copy()
            indices = self.train_mask[task_for_monitoring]
            d["visit"] = self.node_dict["visit"][indices]
            sg = hetero_subgraph(self.graph, d).to(self.device)
            preds_monitor = self.gnn(sg, "visit", task_for_monitoring)
            if task_for_monitoring == "drug_rec":
                all_drugs = self.train_mask["all_drugs"]
                labels = []
                for i in indices:
                    drugs = self.labels[t][i]
                    labels.append([1 if d in drugs else 0 for d in all_drugs])
                labels_monitor = torch.FloatTensor(labels).to(self.device)
                preds_monitor = preds_monitor.sigmoid()
            else:
                labels_monitor = torch.LongTensor([self.labels[task_for_monitoring][i] for i in indices]).to(self.device)

            train_metrics = metrics(preds_monitor, labels_monitor, task_for_monitoring)
            test_metrics = self.evaluate()

            training_range.set_description_str(
                "Epoch {} | loss: {:.4f}| Train ACC: {:.4f} | Test ACC: {:.4f} ".format(
                    epoch, total_loss.item(), train_metrics["train_accuracy"], test_metrics["test_accuracy"])
            )


            epoch_stats.update({"Train Loss": total_loss.item()})
            epoch_stats.update(train_metrics)
            epoch_stats.update(test_metrics)
            if test_metrics["test_accuracy"] > best_test_acc:
                best_test_acc = test_metrics["test_accuracy"]
                model_path = f"./checkpoints/{self.name}_NL{self.config_gnn['num_layers']}_E{self.n_epoch}_{task_for_monitoring}.pt"
                torch.save(self.gnn.state_dict(), model_path)

    def evaluate(self):
        self.gnn.eval()
        test_metrics = {}

        with torch.no_grad():
            for t in self.tasks:
                indices = self.test_mask[t]
                d = self.node_dict.copy()
                d["visit"] = self.node_dict["visit"][indices]
                sg = hetero_subgraph(self.graph, d).to(self.device)

                preds = self.gnn(sg, "visit", t)

                if t == "drug_rec":
                    all_drugs = self.train_mask["all_drugs"]
                    labels = []
                    for i in indices:
                        drugs = self.labels[t][i]
                        labels.append([1 if d in drugs else 0 for d in all_drugs])
                    labels = torch.FloatTensor(labels).to(self.device)
                    preds = preds.sigmoid()
                else:
                    labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

                task_metrics = metrics(preds, labels, t, prefix="test")
                test_metrics.update(task_metrics)

        return test_metrics
    def evaluate_by_task(self, task):
        self.gnn.eval()
        indices = self.test_mask[task]
        if task == "drug_rec":
            all_drugs = self.train_mask["all_drugs"]
            labels = []
            for i in indices:
                drugs = self.labels[task][i]
                labels.append([1 if d in drugs else 0 for d in all_drugs])
            labels = torch.FloatTensor(labels).to(self.device)
        else:
            labels = torch.LongTensor([self.labels[task][i] for i in indices]).to(self.device)
        d = self.node_dict.copy()
        d["visit"] = self.node_dict["visit"][sorted(indices)]
        sg = hetero_subgraph(self.graph, d)
        sg = sg.to(self.device)
        with torch.no_grad():
            preds = self.gnn(sg, "visit", task)
            if task == "drug_rec":
                preds = preds.sigmoid()
        test_metrics = metrics(preds, labels, task, prefix="test")
        return test_metrics

    def get_masks(self, data, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]
        m = {}
        for nt in data.node_types:
            if nt == "visit":
                m[nt] = torch.from_numpy(masks.astype("int32"))
            else:
                m[nt] = torch.zeros(0)
        return m

    def predict_single_test(self, model_path: str, test_idx: int, task: str = "drug_rec", threshold: float = 0.5) -> dict:
        state_dict = torch.load(model_path, map_location=self.device)
        self.gnn.load_state_dict(state_dict)
        self.gnn.to(self.device)
        self.gnn.eval()
        
        if isinstance(test_indices, torch.Tensor):
            test_indices = test_indices.cpu().numpy()
            
        if test_idx < 0 or test_idx >= len(test_indices):
            raise IndexError(f"test_idx {test_idx} is out of range for task '{task}' with {len(test_indices)} examples.")
        
        sample_index = test_indices[test_idx]

        d = self.node_dict.copy()
        d["visit"] = self.node_dict["visit"][torch.tensor([sample_index])]
        
        sg = hetero_subgraph(self.graph, d)
        sg = sg.to(self.device)
        
        with torch.no_grad():
            preds = self.gnn(sg, "visit", task)
        
        probs = torch.sigmoid(preds)
        binary_preds = (probs >= threshold).float()
        
        probs_np = probs.cpu().numpy()
        binary_preds_np = binary_preds.cpu().numpy()
        all_drugs = self.labels["all_drugs"]
        predicted_drug_indices = np.where(binary_preds_np[0] == 1)[0]
        

        predicted_drug_names = []
        for i in predicted_drug_indices:
            drug_code = all_drugs[i]
            predicted_drug_names.append(drug_code)
        
        ground_truth_drug_codes = self.labels["drug_rec"].get(sample_index, [])
            
        return {
            "predicted_probabilities": probs_np,
            "predicted_labels": binary_preds_np,
            "predicted_drug_names": predicted_drug_names,
            "ground_truth_drugs": ground_truth_drug_codes,
            "sample_index": sample_index,
            "ground_truth_drugs_len": len(ground_truth_drug_codes),
        }

    def get_gnn(self, model_path: str) -> dict:
        state_dict = torch.load(model_path, map_location=self.device)
        self.gnn.load_state_dict(state_dict)
        self.gnn.to(self.device)
        return self.gnn
        