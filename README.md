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

## Team
- Cheikh Ahmedou Enaha
- Djilit Abdellahi
- Mohamed Abderhman Nanne
- Mohamed Lemin Taleb Ahmed

## License
[Your chosen license]
