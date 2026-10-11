# 👨‍💻 Academic Engineering Contribution Report

- **Student Name**: Amr Alsayed
- **Student ID**: 2309115638
- **GitHub Username**: github_user
- **Role**: Team Lead & Data Engineer
- **Institution**: Istanbul Gelişim University

---

### Core Responsibilities & Contributions:

1. **Project Leadership & Task Distribution**:
   - Directed the overall project lifecycle, defined academic milestones, and partitioned research responsibilities among team members according to architectural domains.
   - Maintained continuous technical synchronization across model development tracks.

2. **Exploratory Data Analysis (EDA)**:
   - Conducted comprehensive statistical analysis of the raw comment corpus in `01_EDA_and_Visualisation.ipynb`.
   - Generated token length frequency distributions, analyzed uppercase letter intensity, identified class imbalance patterns, and produced unigram/bigram/trigram vocabulary visualizations.

3. **Data Cleaning & Preprocessing Pipeline Development**:
   - Designed and implemented the unified, reproducible NLP data preprocessing engine (`clean_comment_text`).
   - Integrated `emoji.demojize` to transform sentiment-bearing emojis into structured semantic tokens.
   - Built the YouTube slang and informal shorthand normalization mapping (`SLANG_MAP`).
   - Implemented polarity-preserving stopword filtering (strictly preserving critical negation tokens such as `no`, `not`, `never`, `don't`) and morphological lemmatization via `WordNetLemmatizer`.

4. **Repository Architecture & Final Technical Review**:
   - Structured and maintained the project GitHub repository, organizing and standardizing all Jupyter Notebooks (`00` through `06`).
   - Conducted final code audits, verification of execution integrity, reproducibility checks, and validation of all evaluation reports.
