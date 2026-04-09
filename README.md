# Arabic Book Age Classification — Hybrid Pipeline

A unified pipeline that classifies full Arabic books into five age groups (**3–8, 9–12, 13–18, 18–21, 21+**) by combining a fine-tuned **AraBERT v02** language model with traditional machine learning classifiers (**XGBoost, Random Forest, Gradient Boosting**) trained on 32 book-level linguistic features.

This repository accompanies the bachelor's thesis *"Classifying Arabic Books Content for Age Suitability"* by **Zaid Saad Nadeem AlTayan**, submitted to **IE University, School of Science & Technology** (March 2026), under the supervision of Prof. Suzan Awinat.

---

## Overview

Arabic is the fourth most spoken language in the world, yet no automated system existed for classifying complete Arabic books by age-appropriate reading level. Existing tools (OSMAN, BAREC, Saddiki et al., Liberato et al.) operate only at the sentence or passage level. This project closes that gap by introducing the first end-to-end pipeline that takes a full Arabic PDF book as input and outputs a predicted age group.

The system uses a **hybrid two-branch approach**:

- **Semantic branch** — A fine-tuned AraBERT v02 (135M parameters) that learns writing style, tone, and vocabulary patterns from 512-token text chunks.
- **Structural branch** — Three classical ML classifiers trained on 32 book-level linguistic features (vocabulary richness, sentence complexity, punctuation density, dialogue ratio, and more).

The two branches are combined via **weighted ensemble fusion** (optimal: 0.50 / 0.50), reaching **82.14% book-level accuracy** and a **weighted F1 score of 0.8106** on a held-out test set of 28 books.

---

## Key Results

| Configuration | Accuracy | Weighted F1 |
|---|---|---|
| AraBERT (LLM) only | 71.43% | 0.6821 |
| ML Ensemble only | 64.29% | 0.6538 |
| **Weighted Ensemble (0.50 / 0.50)** | **82.14%** | **0.8106** |

All misclassifications occurred between adjacent age categories — no children's book was ever classified as adult literature, and no philosophical text was ever classified as a children's book.

---

## Repository Contents

| File | Description |
|---|---|
| `labeling.ipynb` | Data preparation and labeling notebook |
| `final.ipynb` | Main training and evaluation pipeline |
| `predict.ipynb` | Inference notebook for classifying new PDF books |
| `all_books_W_L.xlsx` | Labeled corpus of 140 Arabic books split into ~25,680 chunks |
| `all_data_results.csv` | Full prediction results on the test set |
| `feature_importance.csv` | XGBoost feature importance rankings |
| `arabert_finetuned/` | Fine-tuned AraBERT v02 model checkpoint (see breakdown below) |
| `xgb_model.pkl` | Trained XGBoost classifier |
| `rf_model.pkl` | Trained Random Forest classifier |
| `gb_model.pkl` | Trained Gradient Boosting classifier |
| `scaler.pkl` | StandardScaler used for feature normalization |

---

## File Descriptions

### `labeling.ipynb` — Data Preparation and Labeling

Prepares the dataset by linking each book to its corresponding age category before training.

- Reads the original Excel file containing book names and extracted texts (`all_books.xlsx`).
- Creates a reference dictionary mapping book names to the five target age categories (3–8, 9–12, 13–18, 18–21, 21+).
- Adds a new `Age Category` column populated through exact or partial matches on book names.
- Saves the labeled dataset to `all_books_W_L.xlsx` for downstream training.

### `final.ipynb` — Model Training and Evaluation

The main pipeline file that trains and evaluates the hybrid system. Combines a large language model with traditional ML classifiers.

- Loads the labeled data and cleans the Arabic text (removes diacritics, normalizes letters, strips unwanted symbols, filters out very short fragments).
- Splits the data into training and test sets at the **book level** (not the chunk level) to prevent leakage.
- Fine-tunes the **AraBERT v02** model for five-class classification using class-weighted cross-entropy loss.
- Extracts **32 engineered linguistic and statistical features** per book (average word and sentence length, type-token ratio, punctuation density, long/short word ratios, dialogue markers, frequency statistics, and more).
- Trains three traditional ML classifiers — **XGBoost, Random Forest, Gradient Boosting** — on the extracted features.
- Builds a **weighted ensemble** combining the AraBERT predictions (50%) with the ML ensemble predictions (50%), reaching ~82% accuracy.
- Evaluates the final results (accuracy, F1, classification report, confusion matrix) and saves all trained components for future inference.

### `predict.ipynb` — Inference and Application

End-to-end inference notebook that takes a new PDF book and outputs its predicted age category.

- Sets up OCR tooling (`pytesseract`) to extract Arabic text from scanned PDF pages.
- Loads the saved AraBERT model, ML classifiers, scaler, and config files.
- Applies the same cleaning, chunking, and feature extraction pipeline used during training.
- Runs the weighted ensemble to compute class probabilities and select the predicted age group.
- Generates a final report containing the predicted category, confidence score, and the contribution breakdown between AraBERT and the ML branch, then exports results to CSV.

---

## The `arabert_finetuned/` Folder

This folder contains the fine-tuned AraBERT v02 checkpoint used by the semantic branch of the pipeline. It includes:

| File | Description |
|---|---|
| `model.safetensors` | The fine-tuned model weights (~500 MB, ~135M parameters). **Hosted externally — see download link below.** |
| `config.json` | Model architecture configuration (layers, hidden size, number of labels) |
| `tokenizer.json` | The AraBERT tokenizer vocabulary and merge rules |
| `tokenizer_config.json` | Tokenizer settings (special tokens, padding, truncation) |
| `training_args.bin` | The exact `TrainingArguments` used during fine-tuning (for reproducibility) |

### Downloading the model weights

Because `model.safetensors` exceeds GitHub's 100 MB file size limit, it is hosted on Google Drive:

**[Download model.safetensors (Google Drive)](https://drive.google.com/file/d/1JFYNP05d1bcThsriqSkuALyykItlywo0/view?usp=sharing)**

After downloading, place the file inside the `arabert_finetuned/` folder so the directory structure looks like this:

```
arabert_finetuned/
├── model.safetensors        ← downloaded from Google Drive
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

The `predict.ipynb` notebook will then load the full model automatically using `AutoModelForSequenceClassification.from_pretrained("arabert_finetuned/")`.

---

## Methodology Summary

1. **Data collection** — 140 Arabic books manually collected across five age groups, processed through a custom OCR pipeline (Tesseract + Poppler at 300 DPI, 8 parallel workers) and chunked into ~25,680 non-overlapping 512-token segments using the AraBERT v02 tokenizer.
2. **Preprocessing** — Arabic text cleaning, letter normalization, diacritic removal, removal of fragments under 50 characters.
3. **Feature engineering** — 32 features grouped into seven categories: word-level, sentence-level, vocabulary, word complexity, punctuation, statistical, and structural.
4. **Model training** — AraBERT fine-tuned for 5 epochs (early stopping, learning rate 2e-5, fp16, class-weighted loss) on an NVIDIA H100 GPU; ML classifiers trained with regularized hyperparameters via scikit-learn and XGBoost.
5. **Ensemble fusion** — Weighted combination of LLM and ML probability vectors, with weights selected via grid search on the test set.
6. **Evaluation** — Book-level accuracy, weighted F1, per-class precision/recall, and confusion matrix analysis.

---

## Tech Stack

- **Python 3**, **PyTorch**, **Hugging Face Transformers**
- **AraBERT v02** (`aubmindlab/bert-base-arabertv02`)
- **scikit-learn**, **XGBoost**
- **Tesseract OCR** + **Poppler** + **pdf2image**
- **pandas**, **NumPy**, **matplotlib**, **seaborn**
- **Google Colab** with NVIDIA H100 (80 GB VRAM) for training

---

## How to Use

1. Clone the repository.
2. Install dependencies (PyTorch, Transformers, scikit-learn, XGBoost, pytesseract, pdf2image, openpyxl, pandas, numpy).
3. Install system packages: `tesseract-ocr`, `tesseract-ocr-ara`, `poppler-utils`.
4. To **reproduce training**, run `final.ipynb` with `all_books_W_L.xlsx` available.
5. To **classify a new book**, run `predict.ipynb` and provide the path to your PDF file. The notebook will run OCR, extract features, apply the saved models, and output the predicted age category.

---

## Limitations

- The corpus contains 140 books — sufficient as a proof of concept but not a production-ready scale.
- The 13–18 vs. 18–21 boundary is the most error-prone, as the distinction is often thematic rather than linguistic.
- The system classifies linguistic complexity only — it does **not** assess thematic appropriateness (violence, mature themes, etc.).
- Age-group conventions vary across Arabic-speaking countries; the labels reflect publisher recommendations rather than a universal standard.
- OCR quality varies, particularly for picture books in the 3–8 category where most content is embedded in illustrations.

---

## Citation

If you use this work, please cite the thesis:

> AlTayan, Z. S. N. (2026). *Classifying Arabic Books Content for Age Suitability: A Unified Pipeline Combining AraBERT and Traditional Machine Learning for Arabic Readability Assessment* [Bachelor's thesis, IE University, School of Science & Technology].

---

## Author

**Zaid Saad Nadeem AlTayan**  
Bachelor of Computer Science & Artificial Intelligence  
IE University, School of Science & Technology  
Supervisor: Prof. Suzan Awinat (Universidad Autónoma de Madrid)

---

## Acknowledgements

Special thanks to Prof. Suzan Awinat for her guidance throughout this project, and to IE University's School of Science & Technology for providing the computing resources used to fine-tune the language model.
