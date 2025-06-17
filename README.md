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
└── ml/             # Machine learning models
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

## Team
- Cheikh Ahmedou Enaha
- Djilit Abdellahi
- Mohamed Abderhman Nanne
- Mohamed Lemin Taleb Ahmed

## License
[Your chosen license]
