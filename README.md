# Advanced Sentiment Analysis with Sarcasm & Emotion Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Interface-Gradio-green)

A state-of-the-art NLP system that combines multiple AI models to detect nuanced sentiment, sarcasm, and 28 distinct emotions in text.

## Table of Contents
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### 🎯 Core Capabilities
- **Multidimensional Sentiment Analysis** (Positive/Neutral/Negative with confidence scores)
- **Sarcasm & Irony Detection** (Specialized model with probability scoring)
- **28-Class Emotion Recognition** (From admiration to grief)
- **Context-Aware Adjustments** (Automatically adapts to sarcastic tones)

### 🚀 Technical Highlights
- **Ensemble Modeling** (5 different NLP models combined)
- **Real-Time Processing** (Optimized for GPU acceleration)
- **Interactive Web Demo** (Gradio interface)
- **Advanced Text Preprocessing** (Lemmatization, stopword removal, etc.)

## Model Architecture

### 🤖 Model Breakdown
| Model | Purpose | Key Strengths |
|-------|---------|---------------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Base sentiment analysis | Fine-tuned on social media data |
| `sismetanin/roberta-base-sarcasm-detection` | Sarcasm identification | Trained on ironic/sarcastic Reddit comments |
| `SamLowe/roberta-base-go_emotions` | Emotion classification | 28 distinct emotional categories |
| `facebook/bart-large-mnli` | Zero-shot classification | Flexible sentiment pattern matching |
| NLTK VADER | Traditional sentiment | Rule-based for lexical patterns |

### 🔄 Processing Pipeline
1. **Text Normalization** (Clean and standardize input)
2. **Parallel Model Inference** (All models process simultaneously)
3. **Score Aggregation** (Weighted ensemble approach)
4. **Context Adjustment** (Sarcasm/emotion-based corrections)
5. **Result Synthesis** (Unified output format)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for best performance)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/enhanced-sentiment-analysis.git
cd enhanced-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download(['vader_lexicon', 'punkt', 'stopwords', 'wordnet'])"
