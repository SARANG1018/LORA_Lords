📰 AG News Classification using LoRA (Low-Rank Adaptation)
This project presents a parameter-efficient fine-tuning pipeline using LoRA on the AG News text classification dataset. The goal is to achieve high accuracy while keeping the trainable parameters under 1 million, making the model efficient for deployment and leaderboard evaluation.

📁 Project Structure
LORA_Lords_Final.ipynb — Main notebook with preprocessing, training, evaluation, and prediction.

test_unlabelled.pkl — Kaggle-provided test dataset (no labels).

inference_output.csv — Final submission predictions for the Kaggle leaderboard.

📌 Highlights
Base Model: roberta-base from Hugging Face.

LoRA Configuration: Applied to query, value layers for SEQ_CLS task.

Total Trainable Parameters: ~0.99M.

Performance:

Validation Accuracy: ~93%

Test Accuracy (Kaggle Leaderboard): ~84.85%

Tools: Optuna, HuggingFace Trainer, PyTorch, PEFT (LoRA), Transformers, Evaluate.

🧠 Key Techniques
Low-Rank Adaptation (LoRA) for lightweight tuning.

Mixed Precision Training (FP16) for faster and memory-efficient training.

Hyperparameter Optimization (Optuna) with smart search space design.

Metrics Tracked: Accuracy, Precision, Recall, F1.

Visualization: All relevant training curves are plotted.

🔧 Best Hyperparameters (via Optuna)
yaml
Copy
Edit
learning_rate: 3.99e-4
batch_size: 32
gradient_accumulation_steps: 4
weight_decay: 0.0
lora_r: 10
lora_alpha: 64
lora_dropout: 0.1
target_modules: ['query', 'value']
📈 Results

Metric	Value
Validation Accuracy	93.0%
Kaggle Test Accuracy	84.85%
Trainable Params	~999,172
Precision	✅ Logged
Recall	✅ Logged
F1 Score	✅ Logged
🧪 All metrics were visualized across steps and validated using both training and held-out test data.

📊 Visualizations
✅ Training vs Evaluation Loss

✅ Train Accuracy (Final) vs Evaluation Accuracy

✅ Precision, Recall, F1 vs Step

All plots are generated using Matplotlib after training using the trainer’s log_history.

🚀 Inference Demo
python
Copy
Edit
classify(model, tokenizer, "Wall St. Bears Claw Back Into the Black")
Predictions are saved to:

bash
Copy
Edit
inference_output_trial9.csv
📚 References
[1] Hu, E., et al. LoRA: Low-Rank Adaptation of Large Language Models, arXiv:2106.09685. [Paper]

[2] He, K., et al. Deep Residual Learning for Image Recognition, CVPR 2016.

[3] AG News Dataset: HuggingFace Datasets

👨‍💻 Team Members
Sarang P. Kadakia (Net ID: sk11634)

Vishwajeet Kulkarni (Net ID: vk2630)
