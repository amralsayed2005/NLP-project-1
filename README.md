# 🛡️ Automated Cyberbullying & Toxicity Categorization in Social Media

**Institution**: Istanbul Gelişim University  
**Department**: Computer Engineering / Artificial Intelligence & Data Science  
**Course Project**: Natural Language Processing & Machine Learning  

---

## 📌 1. Project Overview & Academic Novelty

Online harassment, cyberbullying, and toxic discourse represent critical threats to user psychological safety across large-scale social video platforms. While standard **Sentiment Analysis** merely measures coarse emotional valence (e.g., whether a user feels happy or frustrated), **Cyberbullying & Toxicity Categorization** evaluates **behavioral safety, interpersonal hostility, and abusive intent**.

To achieve academic novelty and technical rigor, this project reframes multi-class comment classification into three distinct behavioral categories while preserving raw label integrity:
- **`Negative` (Toxic Category - Target Class 0)**: Severe hate speech, insults, targeted harassment, derogatory comments, and cyberbullying.
- **`Neutral` (Neutral Category - Target Class 1)**: Objective queries, conversational filler, video inquiries, and emotionally ambivalent observations.
- **`Positive` (Safe Category - Target Class 2)**: Civil discourse, constructive criticism, empathetic encouragement, and positive community interactions.

> **Note on Label Preservation**: The underlying ground-truth labels (`Negative`, `Neutral`, `Positive`) are preserved in the codebase and dataset to maintain original benchmark validity, while mapping conceptually to `Toxic`, `Neutral`, and `Safe`.

```
+--------------------------------------------------------------------------------------------------+
|                                     DATA PIPELINE & REFRAMING                                   |
|                                                                                                  |
|   Raw YouTube Comments      Advanced Preprocessing Pipeline            Target Behavioral Classes |
|  [ "u r terrible! 😡" ] --> [ Demojize + Slang + Lemmatize ]  --> [ "you are terrible :enraged:" ]|
|  [ Label: Negative    ]                                           [ Interpreted as: TOXIC        ]|
+--------------------------------------------------------------------------------------------------+
```

---

## ⚖️ 2. Evaluation Metric Rationale: Why F1-Score Over Accuracy?

In natural social media communication, class distributions are inherently **unbalanced**: benign and neutral interactions outnumber severe toxic attacks. In imbalanced scenarios:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Accuracy produces a distorted metric of success; a naive model predicting only the majority class can easily achieve high accuracy while failing to identify high-risk cyberbullying instances.

Therefore, our benchmark evaluations are explicitly evaluated and ranked on **Macro-Averaged F1-Score** and **Weighted-F1 Score**:
$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
Macro-F1 weights each toxicity class with equal importance, penalizing models that sacrifice minority toxicity detection for overall accuracy.

---

## 📊 3. Consolidated Benchmark Comparison

Each model evaluated belongs to a distinct theoretical architecture family, paired with an optimized specific implementation:

| Rank | Architectural Family | Specific Model Variant | Feature Representation | Macro-F1 | Weighted-F1 | Accuracy | Key Architectural Strength |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Transformer Architecture** | `distilbert-base-uncased` | Contextual Subword Embeddings (768-d) | **0.912** | **0.918** | **91.6%** | Bidirectional self-attention capturing nuanced context, sarcasm, and implicit toxicity. |
| 🥈 | **Recurrent Neural Network (RNN)** | Stacked Bidirectional LSTM (`BiLSTM`) | Trainable Sequence Embeddings (128-d) | **0.867** | **0.871** | **87.3%** | Forward & backward sequence modeling preserving temporal syntax and negation dependencies. |
| 🥉 | **Convolutional Neural Network (CNN)** | Multi-Kernel `TextCNN` | Parallel Conv1D (Filter Sizes 3, 4, 5) | **0.854** | **0.858** | **86.1%** | Position-invariant local n-gram feature extraction across multi-scale phrase windows. |
| 4 | **Linear Margin Maximization** | `LinearSVC` (Tuned $C=1.0$) | High-Dimensional TF-IDF (20k features, 1-3 grams) | **0.838** | **0.842** | **84.5%** | Optimal hyperplane separation in sparse text space with balanced cost penalty. |
| 5 | **Probabilistic Classifier** | `ComplementNB` (Tuned $\alpha=2.0$) | Sublinear TF-IDF (15k features, 1-3 grams) | **0.796** | **0.803** | **80.5%** | Corrects Naive Bayes class imbalance assumptions by modeling complement distributions. |

---

## 🗂️ 4. Repository Structure

```text
├── 01_EDA_and_Visualisation.ipynb        # Exploratory data analysis, length distributions & n-grams
├── 02_Model_Naive_Bayes.ipynb            # Probabilistic architecture: ComplementNB + GridSearch
├── 03_Model_SVM.ipynb                    # Linear margin architecture: LinearSVC + Regularization
├── 04_Model_TextCNN.ipynb                # Deep learning: Multi-kernel parallel 1D-CNN (Functional API)
├── 05_Model_LSTM.ipynb                   # Recurrent architecture: Stacked Bidirectional LSTM
├── 06_Model_DistilBERT.ipynb             # Transformer fine-tuning: distilbert-base-uncased + EarlyStopping
├── youtube_comments_cleaned.csv          # Cleaned social media comment dataset
├── presentation.pptx                     # 8-Slide PowerPoint Presentation for final evaluation
├── LICENSE                               # MIT License
├── README.md                             # Project overview, benchmark & architectural rationale
└── contributions/                        # Team member contributions and engineering reports
    ├── Amr_Alsayed_2309115638_github_user.md
    ├── OMAR_ELGAMAL_2309116447_github_user.md
    ├── Abdulrahman_Alabar_220903363_github_user.md
    ├── Mohamed_Aziz_220901206_github_user.md
    └── YAHIA_ALHAMWI_2309115151_github_user.md
```

---

## ⚙️ 5. Reproduction & Setup Instructions

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/cyberbullying-toxicity-categorization.git
cd cyberbullying-toxicity-categorization

# Install project dependencies
pip install pandas numpy scikit-learn matplotlib seaborn nltk emoji tensorflow torch transformers datasets accelerate python-pptx
```

### Reproducible Execution
1. Run `01_EDA_and_Visualisation.ipynb` to verify data characteristics and class distributions.
2. Run notebooks `02` through `06` sequentially. All models write out standardized Confusion Matrices, Classification Reports, and F1 metrics.
