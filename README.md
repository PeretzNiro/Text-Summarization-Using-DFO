# DFO-Based Text Summarization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![DFO Optimization: Blue](https://img.shields.io/badge/DFO-Optimization-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of extractive text summarization using Dispersive Flies Optimization (DFO) algorithms. This system uses various DFO variants to optimize sentence selection for generating concise and informative summaries of scientific papers.

## Features

- Multiple DFO algorithm variants:
  - Base DFO
  - Elite DFO
  - Momentum DFO
  - Local Search DFO
  - Elite Local DFO
  - Elite Momentum DFO
  - Hybrid (Elite Local Momentum) DFO

- Text preprocessing and feature extraction:
  - Document text cleaning and normalization
  - Sentence segmentation using spaCy
  - Position and length features
  - TF-IDF vectorization
  - Linguistic features (noun phrases, verb phrases, proper nouns)
  - Semantic features:
    - Sentence embeddings
    - Cosine similarity between sentences
    - Document vector similarity
  - Feature normalization
  - Multi-threaded batch processing
  - Efficient dataset loading and caching

- Comprehensive evaluation:
  - ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-L)
  - Statistical analysis of results
  - Performance metrics tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PeretzNiro/Text-Summarization-Using-DFO.git
cd Text-Summarization-Using-DFO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset

The system uses the CNN/DailyMail dataset for training and evaluation. The dataset will be automatically downloaded when you run the code for the first time.

Dataset details:
- Name: CNN/DailyMail
- Version: 3.0.0
- Size: ~11.5k articles for testing
- Source: https://huggingface.co/datasets/cnn_dailymail

## Usage

1. Basic usage:
```python
from src import TextSummarizer, SummarizerConfig

# Create summarizer with default configuration
summarizer = TextSummarizer()

# Generate summary for a document
summary, scores, history = summarizer.summarize_document(document)
```

2. Run with custom configuration:
```python
config = SummarizerConfig(
    algorithm='dfo_elite_local_momentum',
    batch_size=32,
    n_jobs=-1,
    cache_features=True
)
summarizer = TextSummarizer(config)
```

3. Run from command line:
```bash
python main.py --algorithm dfo --num_papers 20 --batch_size 10
```

## Configuration

### DFO Parameters
- population_size: 90
- max_iterations: 100
- delta: 0.6 (exploration factor)
- target_length_ratio: 0.08 (summary length)

### Fitness Calculation Weights
- alpha (Relevance): 0.3
- beta (ROUGE score): 0.7
- gamma (Length penalty): 0.5

## Project Structure

- `main.py`: Entry point and experiment orchestration
- `src/`
  - `summarizer.py`: Main summarization pipeline
  - `preprocessor.py`: Document preprocessing and feature extraction
  - `evaluator.py`: ROUGE evaluation metrics
  - `dfo.py`: Base DFO implementation
  - `dfo_*.py`: DFO variants
  - `utils.py`: Utility functions

## Results

The system generates summaries with the following average performance metrics:
- ROUGE-1: ~0.40 F1-score
- ROUGE-2: ~0.17 F1-score
- ROUGE-3: ~0.15 F1-score
- ROUGE-L: ~0.36 F1-score

Results are saved in:
- `output/results.json`: Detailed evaluation metrics including mean, standard deviation, min, and max scores for each ROUGE metric
- `output/convergence_history.json`: Optimization convergence data

## License
This project is licensed under the [MIT License](LICENSE).