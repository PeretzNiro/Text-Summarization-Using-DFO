# DFO-Based Text Summarization

A Python implementation of extractive text summarization using Dispersive Flies Optimization (DFO) algorithm. This system uses various DFO variants to optimize sentence selection for generating concise and informative summaries of scientific papers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/PeretzNiro/Text-Summarization-Using-DFO?style=social)](https://github.com/PeretzNiro/Text-Summarization-Using-DFO/stargazers)
[![Research](https://img.shields.io/badge/Research-Academic-9cf)](https://github.com/PeretzNiro/Text-Summarization-Using-DFO)
[![NLP](https://img.shields.io/badge/NLP-Text_Summarization-green)](https://github.com/PeretzNiro/Text-Summarization-Using-DFO)
[![Swarm Intelligence](https://img.shields.io/badge/Swarm-Intelligence-purple)](https://github.com/PeretzNiro/Text-Summarization-Using-DFO)
![DFO Optimization: Blue](https://img.shields.io/badge/DFO-Optimization-purple.svg)

## Overview

This project leverages Dispersive Flies Optimization (DFO) to tackle the extractive text summarization problem. DFO is a population-based swarm intelligence algorithm inspired by the behavior of flies hovering over food sources. 

### How It Works

1. **Document Processing**: Each document is processed into sentences and various linguistic features are extracted.
2. **Optimization Problem**: Sentence selection is formulated as an optimization problem where:
   - Each fly in the swarm represents a candidate summary (a subset of sentences)
   - The fitness function balances relevance, coherence, and length constraints
   - The goal is to find the optimal subset of sentences that best represents the document

3. **Algorithm Variants**: Multiple DFO variants have been implemented to improve performance:
   - Elite strategies to preserve the best solutions
   - Local search mechanisms to refine promising solutions
   - Momentum-based updates to escape local optima
   
4. **Evaluation**: Summaries are evaluated using ROUGE metrics against reference summaries.

### Advantages

- **Efficient Exploration**: DFO effectively explores the vast space of possible sentence combinations
- **Parallelizable**: Naturally supports multi-threaded processing for batch summarization
- **Customizable**: Easily adaptable to different document types and summarization requirements
- **No Training Required**: Unlike neural approaches, this method doesn't require supervised training

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