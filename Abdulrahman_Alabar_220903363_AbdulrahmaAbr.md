# 👨‍💻 Academic Engineering Contribution Report

- **Student Name**: Abdulrahman Alabar
- **Student ID**: 220903363
- **GitHub Username**: github_user
- **Role**: Deep Learning Specialist
- **Institution**: Istanbul Gelişim University

---

### Core Responsibilities & Contributions:

1. **Deep Learning Architectural Engineering**:
   - Held primary responsibility for designing, training, regularizing, and benchmarking the project's deep neural network architectures using TensorFlow and Keras.
   - Built sequence tokenization and integer embedding pipelines (`Tokenizer` with 25,000 vocabulary limit, post-sequence padding up to 128 tokens).

2. **Convolutional Neural Network Implementation (`04_Model_TextCNN.ipynb`)**:
   - Engineered the Multi-Kernel TextCNN architecture using the Keras Functional API, replacing single-filter baselines with 3 parallel 1D convolutional branches (kernel sizes 3, 4, and 5) to capture multi-scale local phrase patterns.
   - Integrated `SpatialDropout1D(0.25)`, Batch Normalization after convolutional blocks, and dense Dropout(0.4) with L2 weight regularization (`l2(1e-4)`).
   - Configured dynamic callbacks (`EarlyStopping` monitoring validation loss, `ReduceLROnPlateau`) and passed balanced `class_weight` to combat label imbalance.

3. **Recurrent Neural Network Implementation (`05_Model_LSTM.ipynb`)**:
   - Designed and optimized the Stacked Bidirectional Long Short-Term Memory (`BiLSTM`) network (`Bidirectional(LSTM(128))` followed by `Bidirectional(LSTM(64))`) to model bidirectional temporal syntax and long-range dependencies.
   - Implemented recurrent dropout, spatial dropout, Batch Normalization, and early stopping callbacks restoring optimal validation weights.
   - Generated loss/accuracy training trajectories, confusion matrices, and multi-class classification reports.
