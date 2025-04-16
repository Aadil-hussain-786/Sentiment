import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from torch.nn.functional import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
import gradio as gr
from typing import Dict, Tuple

# Ensure NLTK resources are downloaded
nltk.download(['vader_lexicon', 'punkt'])

# Suppress warnings
warnings.filterwarnings('ignore')

class AdvancedSentimentAnalyzer:
    """
    Advanced Sentiment Analysis Tool with multiple analysis methods
    """
    def __init__(self):
        # Initialize the transformer-based model (fine-tuned for sentiment analysis)
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Initialize the zero-shot classifier for broader sentiment analysis
        self.zero_shot_pipeline = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli")
        
        # Initialize traditional NLP sentiment analyzer (VADER)
        self.sia = SentimentIntensityAnalyzer()
        
        # Configure device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Sentiment labels mapping
        self.id2label = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        text = text.strip()
        return text
    
    def analyze_with_transformers(self, text: str) -> Dict:
        """Analyze sentiment using fine-tuned transformer model"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            scores = outputs.logits[0].detach().cpu()
            scores = softmax(scores, dim=0).numpy()
            
            sentiment_scores = {
                self.id2label[i]: float(score) 
                for i, score in enumerate(scores)
            }
            
            return sentiment_scores
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return {"Error": str(e)}
    
    def analyze_with_zeroshot(self, text: str) -> Dict:
        """Analyze sentiment using zero-shot classification"""
        try:
            candidate_labels = ["positive", "negative", "neutral"]
            result = self.zero_shot_pipeline(text, candidate_labels)
            return {label: score for label, score in zip(result['labels'], result['scores'])}
        except Exception as e:
            print(f"Zero-shot analysis failed: {e}")
            return {"Error": str(e)}
    
    def analyze_with_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER sentiment analyzer"""
        try:
            scores = self.sia.polarity_scores(text)
            return {
                "Negative": scores['neg'],
                "Neutral": scores['neu'],
                "Positive": scores['pos']
            }
        except Exception as e:
            print(f"VADER analysis failed: {e}")
            return {"Error": str(e)}
    
    def ensemble_analysis(self, text: str) -> Dict:
        """Combine results from multiple methods for robust analysis"""
        text = self._preprocess_text(text)
        
        # Get results from all methods
        transformer_results = self.analyze_with_transformers(text)
        zeroshot_results = self.analyze_with_zeroshot(text)
        vader_results = self.analyze_with_vader(text)
        
        # Normalize zero-shot results to match other formats
        normalized_zeroshot = {
            "Negative": zeroshot_results.get("negative", 0),
            "Neutral": zeroshot_results.get("neutral", 0),
            "Positive": zeroshot_results.get("positive", 0)
        }
        
        # Calculate weighted average (giving more weight to transformer model)
        ensemble_scores = {
            "Negative": (0.5 * transformer_results["Negative"] + 
                        0.3 * normalized_zeroshot["Negative"] + 
                        0.2 * vader_results["Negative"]),
            "Neutral": (0.5 * transformer_results["Neutral"] + 
                       0.3 * normalized_zeroshot["Neutral"] + 
                       0.2 * vader_results["Neutral"]),
            "Positive": (0.5 * transformer_results["Positive"] + 
                       0.3 * normalized_zeroshot["Positive"] + 
                       0.2 * vader_results["Positive"])
        }
        
        # Normalize to ensure sum to 1
        total = sum(ensemble_scores.values())
        ensemble_scores = {k: v/total for k, v in ensemble_scores.items()}
        
        return {
            "Transformer": transformer_results,
            "ZeroShot": normalized_zeroshot,
            "VADER": vader_results,
            "Ensemble": ensemble_scores,
            "Final_Sentiment": max(ensemble_scores.items(), key=lambda x: x[1])[0]
        }

def visualize_results(results: Dict) -> Tuple[str, Dict]:
    """Create visual representation of the analysis results"""
    final_sentiment = results["Final_Sentiment"]
    ensemble_scores = results["Ensemble"]
    
    # Create sentiment emoji
    sentiment_emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }.get(final_sentiment, "🤔")
    
    # Create markdown output
    markdown_output = f"""
    ## Sentiment Analysis Results
    
    **Final Sentiment**: {final_sentiment} {sentiment_emoji}
    
    ### Confidence Scores:
    - Positive: {ensemble_scores['Positive']:.2%}
    - Neutral: {ensemble_scores['Neutral']:.2%}
    - Negative: {ensemble_scores['Negative']:.2%}
    
    ### Detailed Breakdown:
    **Transformer Model (RoBERTa):**
    - Positive: {results['Transformer']['Positive']:.2%}
    - Neutral: {results['Transformer']['Neutral']:.2%}
    - Negative: {results['Transformer']['Negative']:.2%}
    
    **Zero-Shot Classification:**
    - Positive: {results['ZeroShot']['Positive']:.2%}
    - Neutral: {results['ZeroShot']['Neutral']:.2%}
    - Negative: {results['ZeroShot']['Negative']:.2%}
    
    **VADER Sentiment Analyzer:**
    - Positive: {results['VADER']['Positive']:.2%}
    - Neutral: {results['VADER']['Neutral']:.2%}
    - Negative: {results['VADER']['Negative']:.2%}
    """
    
    return markdown_output, ensemble_scores

# Initialize analyzer
analyzer = AdvancedSentimentAnalyzer()

def analyze_sentiment(text: str) -> Dict:
    """Main function to analyze sentiment of input text"""
    if not text.strip():
        return {"error": "Please enter some text to analyze."}
    
    results = analyzer.ensemble_analysis(text)
    markdown_output, scores = visualize_results(results)
    
    return {
        "markdown": markdown_output,
        "scores": scores,
        "sentiment": results["Final_Sentiment"]
    }

# Create Gradio interface
with gr.Blocks(title="Advanced Sentiment Analysis Tool", theme="soft") as demo:
    gr.Markdown("# 🌟 Advanced AI Sentiment Analysis Tool")
    gr.Markdown("Analyze text sentiment using state-of-the-art NLP models")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter your text here",
                placeholder="Type or paste text to analyze sentiment...",
                lines=5
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
        with gr.Column():
            output_markdown = gr.Markdown(label="Analysis Results")
            sentiment_output = gr.Label(label="Predicted Sentiment")
            with gr.Accordion("Detailed Scores", open=False):
                sentiment_scores = gr.Label(label="Confidence Scores")
    
    # Examples
    gr.Examples(
        examples=[
            ["I absolutely love this product! It's amazing and works perfectly."],
            ["The service was terrible and the staff was rude. Never coming back!"],
            ["The meeting was scheduled for 2pm in the conference room."],
            ["This movie was okay, nothing special but not bad either."]
        ],
        inputs=input_text,
        label="Try these examples"
    )
    
    # Analysis function
    def analyze_and_display(text):
        result = analyze_sentiment(text)
        return {
            output_markdown: result["markdown"],
            sentiment_output: result["sentiment"],
            sentiment_scores: result["scores"]
        }
    
    analyze_btn.click(
        fn=analyze_and_display,
        inputs=input_text,
        outputs=[output_markdown, sentiment_output, sentiment_scores]
    )

if __name__ == "__main__":
    # Run the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=7860)
