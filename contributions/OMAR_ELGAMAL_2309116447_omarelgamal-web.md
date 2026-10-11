# 👨‍💻 Academic Engineering Contribution Report

- **Student Name**: OMAR ELGAMAL
- **Student ID**: 2309116447
- **GitHub Username**: omarelgamal-web
- **Role**: Classical Machine Learning Specialist

---

### Core Responsibilities & Contributions:

1. **End-to-End Classical ML Pipeline Ownership**:
   - Held sole responsibility for the conception, implementation, hyperparameter optimization, and evaluation of all classical machine learning models.
   - Built the feature extraction pipeline utilizing high-dimensional `TfidfVectorizer` (sublinear term frequency scaling, expanded unigram-to-trigram feature spaces, and document frequency filtering).

2. **Probabilistic Architecture Implementation (`02_Model_Naive_Bayes.ipynb`)**:
   - Implemented both `MultinomialNB` and `ComplementNB`, specifically evaluating `ComplementNB` to address the empirical class imbalance present in social media comments.
   - Designed and executed 5-fold cross-validated `GridSearchCV` over smoothing parameter $\alpha \in [0.01, 5.0]$ and prior fitting strategies.
   - Generated standardized Confusion Matrix heatmaps, classification reports, and qualitative error analysis of misclassified samples.

3. **Linear Margin Maximization Implementation (`03_Model_SVM.ipynb`)**:
   - Engineered the `LinearSVC` classification model on a 20,000-dimensional TF-IDF feature space.
   - Tuned regularization penalties ($C \in [0.01, 10.0]$), loss criteria (`hinge` vs. `squared_hinge`), and incorporated cost-sensitive class weights (`class_weight='balanced'`).
   - Extracted and visualized top positive and negative linear decision boundary coefficients per class for model interpretability.
