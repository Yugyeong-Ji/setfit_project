import json
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset


with open("src/config.json", "r") as f:
    config = json.load(f)

dataset = load_dataset(config["dataset_name"])
label_set = list(set(dataset["train"]["label"]))
few_shot = []
per_label_count = {label: 0 for label in label_set}

for ex in dataset["train"]:
    if per_label_count[ex["label"]] < config["num_shots_per_class"]:
        few_shot.append(ex)
        per_label_count[ex["label"]] += 1
    if all(count >= config["num_shots_per_class"] for count in per_label_count.values()):
        break

# Convert to SetFit format
# train_texts = [ex["sentence"] for ex in few_shot]
# train_labels = [ex["label"] for ex in few_shot]

# train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
train_dataset = Dataset.from_dict({
    "text": [ex["sentence"] for ex in few_shot],
    "label": [ex["label"] for ex in few_shot]
})
eval_dataset = Dataset.from_dict({
    "text": dataset["validation"]["sentence"],
    "label": dataset["validation"]["label"]
})

# Load model
model = SetFitModel.from_pretrained(config["model_name"])

# Trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    batch_size=config["batch_size"],
    num_iterations=config["num_iterations"]
)

trainer.train()
metrics = trainer.evaluate()

print("Accuracy:", metrics["accuracy"])
print("F1 Score:", f1_score(dataset["validation"]["label"], trainer.model.predict(dataset["validation"]["sentence"]), average="weighted"))
