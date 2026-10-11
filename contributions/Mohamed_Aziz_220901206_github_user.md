# 👨‍💻 Academic Engineering Contribution Report

- **Student Name**: Mohamed Aziz
- **Student ID**: 220901206
- **GitHub Username**: ------
- **Role**: Transformer Architecture Specialist


---

### Core Responsibilities & Contributions:

1. **State-of-the-Art Transformer Fine-Tuning (`06_Model_DistilBERT.ipynb`)**:
   - Exclusively held ownership of the project's state-of-the-art transformer architecture, utilizing Hugging Face `transformers`, `datasets`, and PyTorch.
   - Designed and executed the sequence classification pipeline for `distilbert-base-uncased` across the multi-class social media comment dataset.

2. **Tokenizer Pipeline & Subword Feature Processing**:
   - Implemented `AutoTokenizer` tokenization with dynamic batching, truncation, and padding to uniform sequence lengths of 128 subwords.
   - Formatted inputs into PyTorch tensor datasets with attention masks to capture bidirectional contextual representations.

3. **Hyperparameter Optimization & TrainingArguments Engineering**:
   - Formulated optimal Hugging Face `TrainingArguments`: configured a Cosine learning rate scheduler with 10% warmup steps, learning rate of $3\times 10^{-5}$, and weight decay of $0.01$.
   - Configured step-level evaluation checkpoints (`eval_steps=100`, `save_steps=100`) and integrated `EarlyStoppingCallback(early_stopping_patience=3)` monitored on validation F1-score to prevent overfitting.
   - Enabled `fp16` mixed-precision execution for accelerated tensor computations.

4. **Evaluation & Project Benchmark Leadership**:
   - Evaluated fine-tuned model predictions, generating test set Accuracy, macro-averaged and weighted classification reports, and styled confusion matrix heatmaps.
   - Achieved the project's highest benchmark performance (0.912 Macro-F1 / 91.6% Accuracy), confirming the supremacy of self-attention for subtle text categorization.
