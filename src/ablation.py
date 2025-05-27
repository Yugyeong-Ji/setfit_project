# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py


import random
import numpy as np
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from statistics import mean

# config
model_name = "paraphrase-mpnet-base-v2"
num_shots_per_class = 8
batch_size = 64
num_iterations = 20
num_epochs = 1
learning_rate = 1e-3
num_seeds = 10
dataset_name = "sst2"

# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# few-shot sampler
def sample_few_shot(dataset, seed):
    label_set = list(set(dataset["label"]))
    per_label_count = {label: 0 for label in label_set}
    few_shot = []
    random.seed(seed)
    data = list(dataset)
    random.shuffle(data)
    for ex in data:
        if per_label_count[ex["label"]] < num_shots_per_class:
            few_shot.append(ex)
            per_label_count[ex["label"]] += 1
        if all(c >= num_shots_per_class for c in per_label_count.values()):
            break
    return few_shot

# load dataset
raw_dataset = load_dataset("glue", dataset_name)
train_data = raw_dataset["train"]
val_data = raw_dataset["validation"]

# Ablation experiment
accs, f1s = [], []

for seed in range(num_seeds):
    print(f"[Ablation - Seed {seed}]")
    set_seed(seed)

    # sample few-shot data
    few_shot = sample_few_shot(train_data, seed)

    # Load pre-trained model w/o contrastive fine-tuning
    model = SentenceTransformer(model_name)

    # Directly train classifier on frozen embeddings
    X_train = model.encode([ex["sentence"] for ex in few_shot])
    y_train = [ex["label"] for ex in few_shot]
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=seed
    )
    clf.fit(X_train, y_train)

    # Evaluate
    X_val = model.encode(val_data["sentence"])
    y_val = val_data["label"]
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    accs.append(acc)
    f1s.append(f1)

    print(f"[Ablation - Seed {seed}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

print("\nFinal Results (Ablation: No Contrastive Fine-Tuning)")
print(f"Avg Accuracy: {round(mean(accs), 4)}")
print(f"Avg F1 Score: {round(mean(f1s), 4)}")
