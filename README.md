🔍 AG News Classification using Parameter-Efficient Fine-Tuning (LoRA)
This project showcases the use of Low-Rank Adaptation (LoRA) for fine-tuning a pre-trained transformer (RoBERTa) on the AG News dataset. The goal is to achieve high classification accuracy with a model that is both efficient and lightweight — keeping trainable parameters under 1 million.

📁 Project Structure
LORA_Lords_Final.ipynb – Main training notebook with hyperparameter tuning, evaluation, and inference.

test_unlabelled.pkl – Custom test dataset provided via Kaggle competition.

inference_output.csv – Predictions generated for the Kaggle leaderboard submission.

📌 Highlights
Model Base: roberta-base from Hugging Face Transformers.

Fine-Tuning Strategy: LoRA (Low-Rank Adaptation) for SEQ_CLS tasks.

Trainable Parameters: ~0.99M (under 1M) using LoRA with selective layer updates.

Evaluation Metric: Accuracy, Precision, Recall, F1.

Final Accuracy:

Validation Accuracy: ~93%

Kaggle Private Leaderboard Accuracy: ~84.85%

🧠 Techniques Used
Low-Rank Adaptation (LoRA) for efficient fine-tuning

Quantization Ready (optional via BitsAndBytesConfig)

Optuna-based Hyperparameter Search

Learning rate, batch size, gradient accumulation steps, weight decay

Training Strategy

Evaluation during training (every 400 steps)

Early stopping callback

Mixed-precision training (FP16)

Final inference on Kaggle-provided test set

⚙️ Best Hyperparameters (via Optuna)
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
Eval Acc	~93%
Test Acc (Kaggle)	~84.85%
Params (Trainable)	~0.99M
Precision	✅ Logged
Recall	✅ Logged
F1 Score	✅ Logged
📊 Visualizations
The notebook includes:

Training vs Evaluation Loss

Training vs Evaluation Accuracy

Validation Precision / Recall / F1 vs Step

Final Test Accuracy printed post-training

🚀 Inference
python
Copy
Edit
classify(final_model, tokenizer, "Example News Headline")
Predictions are written to inference_output.csv.

📚 References
[1] Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.

[3] AG News Dataset: Hugging Face Datasets

👨‍💻 Team
Sarang P. Kadakia (Net ID: sk11634)

Vishwajeet Kulkarni (Net ID: vk2630)

Rujuta Joshi (Net ID: rj2719)
