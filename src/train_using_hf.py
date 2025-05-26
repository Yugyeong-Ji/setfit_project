import json
from datasets import load_dataset
from setfit import SetFitModel, Trainer
from sklearn.metrics import accuracy_score, f1_score

from datasets import load_dataset, Dataset

import random
import numpy as np
import os

# from setfit import SetFitModel, SetFitTrainer
from statistics import mean
import torch


# # DEBUG
# print(SetFitTrainer.__module__)
# print(SetFitTrainer.__init__.__code__)


# add random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# metric
def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

with open("config.json", "r") as f:
    config = json.load(f)

dataset = load_dataset(config["dataset_name"])
label_set = list(set(dataset["train"]["label"]))

all_accuracies = []
all_f1s = []

for seed in range(10):
    print(f"[Seed {seed}]")
    set_seed(seed)

    # Few-shot sampling
    few_shot = []
    per_label_count = {label: 0 for label in label_set}
    shuffled_data = list(dataset["train"])
    random.shuffle(shuffled_data)

    for ex in shuffled_data:
        if per_label_count[ex["label"]] < config["num_shots_per_class"]:
            few_shot.append(ex)
            per_label_count[ex["label"]] += 1
        if all(count >= config["num_shots_per_class"] for count in per_label_count.values()):
            break

    train_dataset = Dataset.from_dict({
        "text": [ex["sentence"] for ex in few_shot],
        "label": [ex["label"] for ex in few_shot]
    })

    eval_dataset = Dataset.from_dict({
        "text": dataset["validation"]["sentence"],
        "label": dataset["validation"]["label"]
    })

    model = SetFitModel.from_pretrained(config["model_name"])

    trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric=compute_metrics
    )
    # step 1: Contrastive fine-tuning
    trainer.train(
        batch_size=config["batch_size"],
        num_iterations=config["num_iterations"],
        distance_metric="cosine"
    )
    # step 2: Classifier fine-tuning
    # model.model_head.fit(train_dataset["text"], train_dataset["label"])
    X_train = model.encode(train_dataset["text"])
    y_train = train_dataset["label"]
    model.model_head.fit(X_train, y_train)

    # trainer.train()
    metrics = trainer.evaluate()

    print("Accuracy:", metrics["accuracy"])
    print("F1 Score:", metrics["f1"])

    all_accuracies.append(metrics["accuracy"])
    all_f1s.append(metrics["f1"])

# averaged result
print("\n Final Results over 10 seeds")
print("Avg Accuracy:", round(mean(all_accuracies), 4))
print("Avg F1 Score:", round(mean(all_f1s), 4))
