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
  --output-dir ./ml_models/trained_model
```

The script expects a CSV file with a `text` column and a `label` column
(alternatively `generated` or `is_ai_generated`). The trained model and
tokenizer will be saved to the directory specified by `--output-dir`.

## Training the Image Model

Use `ml/train_smogy.py` to fine-tune the Smogy image classifier. The script
expects a dataset in the `imagefolder` format with `train` and optional
`validation` splits.

```bash
python ml/train_smogy.py \
  --dataset-path path/to/images \
  --output-dir ./models/smogy
```

## Exporting the Model to ONNX

After training you can export the model to ONNX format using `ml/export_to_onnx.py`.

```bash
python ml/export_to_onnx.py \
  --model-dir ./ml_models/trained_model \
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
