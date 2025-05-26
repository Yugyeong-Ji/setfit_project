# SetFit Few-Shot Text Classification
---
### 🧪 Training Methods

This project provides two implementations for training SetFit models:

1. **`src/train.py`**

   * Implements the original SetFit training pipeline **directly using `sentence-transformers`**.
   * This version **faithfully reproduces the experiments described in the SetFit paper**, including contrastive fine-tuning and training a logistic regression classifier on sentence embeddings.

2. **`src/train_using_hf.py`**

   * Implements the SetFit pipeline using the **Hugging Face `setfit` library** and its built-in `Trainer`.
   * This version provides a more convenient abstraction for quick experimentation while maintaining the same underlying training principles.

---

### 🚀 How to Run

If you are using an **RTX 4000 series GPU**, you must disable NCCL P2P and IB communication manually due to compatibility issues with `accelerate`:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py
```

For all other GPUs, you can run the training script normally:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

---

### ⚙️ Experiment Configuration

All experiments were run using the **same setup described in the SetFit paper**, including architecture, training strategy, and hyperparameters:

```python
# config (paper-aligned)
model_name = "paraphrase-mpnet-base-v2"
num_shots_per_class = 8
batch_size = 64
num_iterations = 20
num_epochs = 1
learning_rate = 1e-3
num_seeds = 10
dataset_name = "sst2"
```

---

### 📊 Performance Comparison (SST-2, 8-shot)

| Metric                  | Our Implementation (Reproduced) | SetFit Paper   |
| ----------------------- | ------------------------------- | -------------- |
| Avg Accuracy (10 seeds) | **84.21%**                      | **84.0%**      |
| Avg F1 Score (10 seeds) | **84.09%**                      | *Not reported* |

✅ Our implementation **faithfully reproduces** the performance of the SetFit paper on the SST-2 dataset using 8-shot contrastive fine-tuning and logistic regression on top of the fine-tuned embeddings.
