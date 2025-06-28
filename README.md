# Authentiscore

Authentiscore is an AI-powered tool designed to detect AI-generated text content. The system provides probability scores based on various language models (GPT-2, GPT-3, LLaMA 2, Claude 2).

## Project Overview

### Key Features
- Web interface for text analysis
- Real-time AI content detection
- Probability scoring system
- User feedback collection
- Performance metrics dashboard (Accuracy, Recall, Precision, F1-Score, AUC)

### Tech Stack
- Frontend: Next.js
- Backend: FastAPI
- ML Models: Various LLMs (GPT-2, etc.)
- Database: (Your chosen DB)
- Infrastructure: AWS/Azure (planned)

## Project Structure
```
authentiscore/
├── submodules/
│   ├── dbt/         # Data transformation
│   └── airflow/     # Data pipeline orchestration
├── frontend/        # Next.js web interface
├── backend/         # FastAPI server
├── ml/             # Machine learning models
└── models/
    └── smogy/       # Pretrained Smogy model
```

## Getting Started

### Prerequisites
- Git
- Python 3.x
- Node.js

### Cloning the Repository

#### Using SSH
If you have your SSH key set up, you can clone the repository (with all submodules) using:

```bash
git clone --recurse-submodules git@github.com:enaha76/authenti-score.git
```

#### Using HTTPS
If you prefer to clone using HTTPS, run:

```bash
git clone --recurse-submodules https://github.com/enaha76/authenti-score.git
```

If you've already cloned the repository without the `--recurse-submodules` flag, initialize and update the submodules with:

```bash
git submodule update --init --recursive
```

## Training the Model

The `ml/train_model.py` script replicates the notebook steps for preparing the
dataset and fine-tuning a text classification model. You can provide the
dataset path, pretrained model name, and output directory via command-line
arguments or environment variables.

```bash
# Example
python ml/train_model.py \
  --dataset-path path/to/AI_Human.csv \
  --model-name distilbert-base-uncased \
  --output-dir ./ml_models/trained_models
```

The script expects a CSV file with a `text` column and a `label` column
(alternatively `generated` or `is_ai_generated`). A new subdirectory named
`run_YYYYMMDD_HHMMSS` will be created inside the specified `--output-dir` and
the trained model along with its tokenizer will be saved there.
You can also pass `--save-train-texts train_texts.csv` to store the texts used
for fine-tuning. Later, use this file with the `--exclude-path` option when
evaluating to avoid overlap.

### Creating Train/Validation/Test Splits

To make sure your evaluation set is independent from the data used for training
and validation, first split the dataset into three parts. Below is an example
using `scikit‑learn`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("AI_Human.csv")
train_val, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val["label"])

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
```

Use `train.csv` (and `val.csv`) when running `ml/train_model.py` and keep
`test.csv` for evaluation.

## Evaluating the Model

After fine‑tuning you can evaluate the model on your held‑out test split using
`ml/evaluate_model.py`. The `--exclude-path` option removes any samples that
also appear in your training/validation data.

```bash
python ml/evaluate_model.py \
  --dataset-path test.csv \
  --model-dir ./ml_models/trained_models/run_YYYYMMDD_HHMMSS \
  --exclude-path train_texts.csv \
  --sample-size 0.2  # evaluate on 20% of the test set
```

The `--sample-size` option accepts either a fraction (between 0 and 1) or an
integer number of samples to evaluate.

## Training the Image Model

Use `ml/train_smogy.py` to fine-tune the Smogy image classifier. The script
expects a dataset in the `imagefolder` format with `train` and optional
`validation` splits.

```bash
python ml/train_smogy.py \
  --dataset-path path/to/images \
  --output-dir ./models/smogy \
  --train-size 1000  # optional
```

Use `--train-size` and `--val-size` to limit the number of training and
validation samples if you want to train on only a portion of the dataset.

## Exporting the Model to ONNX

After training you can export the model to ONNX format using `ml/export_to_onnx.py`.

```bash
python ml/export_to_onnx.py \
  --model-dir ./ml_models/trained_models/run_YYYYMMDD_HHMMSS \
  --output-path ./ml_models/model.onnx
```

This script loads the saved model and tokenizer, exports it using opset version 14, and verifies the resulting ONNX file.


## Smogy Image Model

### Downloading the Model

Use `scripts/download_smogy.py` to download the pretrained Smogy image classifier from Hugging Face.

```bash
python scripts/download_smogy.py --output-dir models/smogy
```

### Running Inference

The `scripts/smogy_inference.py` script runs the classifier on a single image.

```bash
python scripts/smogy_inference.py path/to/image.jpg --model-dir models/smogy
```

### Fine-tuning

To fine‑tune the model on your own dataset, use `scripts/train_smogy.py`.

```bash
python scripts/train_smogy.py \
  --dataset-path path/to/images \
  --output-dir models/smogy
```

## API Usage

### `POST /predict-image`

Send an image file using `multipart/form-data` or provide a base64 string using the `image_base64` form field.

```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@example.jpg"
```

The response contains the predicted label and confidence score.

If the image includes C2PA metadata from any generator, the API responds with
`"AI-generated (watermark detected)"` and, when available, returns the
generator name without running the classifier.

If no watermark is found, the Smogy model is used for classification. Example response:

```json
{
  "prediction": "Real",
  "is_ai_generated": false,
  "confidence": 0.97
}
```


## Team
- Cheikh Ahmedou Enaha
- Djilit Abdellahi
- Mohamed Abderhman Nanne
- Mohamed Lemin Taleb Ahmed

## License
[Your chosen license]
