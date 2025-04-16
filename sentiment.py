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
from typing import Dict, Tuple, List
import re
from collections import defaultdict

# Ensure NLTK resources are downloaded
nltk.download(['vader_lexicon', 'punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress warnings
warnings.filterwarnings('ignore')

class EnhancedSentimentAnalyzer:
    """
    Advanced Sentiment Analysis Tool with sarcasm and emotion detection
    """
    def __init__(self):
        # Initialize the transformer-based sentiment model
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        
        # Initialize sarcasm detection model
        self.sarcasm_model_name = "sismetanin/roberta-base-sarcasm-detection"
        self.sarcasm_tokenizer = AutoTokenizer.from_pretrained(self.sarcasm_model_name)
        self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained(self.sarcasm_model_name)
        
        # Initialize emotion detection model
        self.emotion_model_name = "SamLowe/roberta-base-go_emotions"
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.emotion_model_name)
        
        # Initialize zero-shot classifier
        self.zero_shot_pipeline = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli")
        
        # Initialize traditional NLP sentiment analyzer (VADER)
        self.sia = SentimentIntensityAnalyzer()
        
        # Configure device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model.to(self.device)
        self.sarcasm_model.to(self.device)
        self.emotion_model.to(self.device)
        
        # Sentiment labels mapping
        self.id2label = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        
        # Emotion labels (from the Go Emotions dataset)
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        # Text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#' from hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation and special chars
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords and lemmatize
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words])
        
        return text.strip()
    
    def detect_sarcasm(self, text: str) -> Dict:
        """Detect sarcasm in text"""
        try:
            inputs = self.sarcasm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sarcasm_model(**inputs)
            
            scores = outputs.logits[0].detach().cpu()
            scores = softmax(scores, dim=0).numpy()
            
            return {
                "Not Sarcastic": float(scores[0]),
                "Sarcastic": float(scores[1])
            }
        except Exception as e:
            print(f"Sarcasm detection failed: {e}")
            return {"Error": str(e)}
    
    def analyze_emotions(self, text: str, top_k: int = 3) -> Dict:
        """Analyze emotions in text"""
        try:
            inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
            
            scores = outputs.logits[0].detach().cpu()
            scores = softmax(scores, dim=0).numpy()
            
            # Get top emotions
            top_indices = np.argsort(scores)[-top_k:][::-1]
            top_emotions = {
                self.emotion_labels[i]: float(scores[i])
                for i in top_indices
            }
            
            return top_emotions
        except Exception as e:
            print(f"Emotion analysis failed: {e}")
            return {"Error": str(e)}
    
    def analyze_with_transformers(self, text: str) -> Dict:
        """Analyze sentiment using fine-tuned transformer model"""
        try:
            inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            
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
            candidate_labels = ["positive", "negative", "neutral", "sarcastic", "ironic"]
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
                "Positive": scores['pos'],
                "Compound": scores['compound']
            }
        except Exception as e:
            print(f"VADER analysis failed: {e}")
            return {"Error": str(e)}
    
    def _adjust_for_sarcasm(self, sentiment_scores: Dict, sarcasm_prob: float) -> Dict:
        """Adjust sentiment scores based on sarcasm probability"""
        if sarcasm_prob > 0.7:  # High probability of sarcasm
            # Invert the sentiment scores
            adjusted_scores = {
                "Positive": sentiment_scores["Negative"] * sarcasm_prob,
                "Neutral": sentiment_scores["Neutral"] * (1 - sarcasm_prob/2),
                "Negative": sentiment_scores["Positive"] * sarcasm_prob
            }
            # Normalize
            total = sum(adjusted_scores.values())
            return {k: v/total for k, v in adjusted_scores.items()}
        return sentiment_scores
    
    def _adjust_for_emotions(self, sentiment_scores: Dict, emotions: Dict) -> Dict:
        """Adjust sentiment based on detected emotions"""
        negative_emotions = {'anger', 'annoyance', 'disappointment', 'disapproval', 
                           'disgust', 'embarrassment', 'fear', 'grief', 'remorse', 
                           'sadness', 'confusion'}
        
        positive_emotions = {'admiration', 'amusement', 'approval', 'caring', 
                           'excitement', 'gratitude', 'joy', 'love', 'optimism', 
                           'pride', 'relief', 'surprise'}
        
        # Calculate emotion influence
        neg_influence = sum(score for emo, score in emotions.items() if emo in negative_emotions)
        pos_influence = sum(score for emo, score in emotions.items() if emo in positive_emotions)
        
        # Adjust sentiment scores
        if neg_influence > pos_influence:
            adjustment_factor = neg_influence - pos_influence
            sentiment_scores["Negative"] = min(1.0, sentiment_scores["Negative"] + adjustment_factor)
            sentiment_scores["Positive"] = max(0.0, sentiment_scores["Positive"] - adjustment_factor/2)
        elif pos_influence > neg_influence:
            adjustment_factor = pos_influence - neg_influence
            sentiment_scores["Positive"] = min(1.0, sentiment_scores["Positive"] + adjustment_factor)
            sentiment_scores["Negative"] = max(0.0, sentiment_scores["Negative"] - adjustment_factor/2)
        
        # Normalize
        total = sum(sentiment_scores.values())
        return {k: v/total for k, v in sentiment_scores.items()}
    
    def ensemble_analysis(self, text: str) -> Dict:
        """Combine results from multiple methods for robust analysis"""
        text = self._preprocess_text(text)
        
        if not text.strip():
            return {"error": "Text is empty after preprocessing"}
        
        # Get results from all methods
        transformer_results = self.analyze_with_transformers(text)
        zeroshot_results = self.analyze_with_zeroshot(text)
        vader_results = self.analyze_with_vader(text)
        sarcasm_results = self.detect_sarcasm(text)
        emotion_results = self.analyze_emotions(text)
        
        # Normalize zero-shot results to match other formats
        normalized_zeroshot = {
            "Negative": zeroshot_results.get("negative", 0) + zeroshot_results.get("ironic", 0)/2,
            "Neutral": zeroshot_results.get("neutral", 0),
            "Positive": zeroshot_results.get("positive", 0),
            "Sarcastic": zeroshot_results.get("sarcastic", 0)
        }
        
        # Adjust transformer results for sarcasm
        sarcasm_prob = max(sarcasm_results.get("Sarcastic", 0), normalized_zeroshot.get("Sarcastic", 0))
        adjusted_transformer = self._adjust_for_sarcasm(transformer_results, sarcasm_prob)
        
        # Further adjust for emotions
        final_sentiment = self._adjust_for_emotions(adjusted_transformer, emotion_results)
        
        # Calculate weighted average
        ensemble_scores = {
            "Negative": (0.4 * final_sentiment["Negative"] + 
                        0.3 * normalized_zeroshot["Negative"] + 
                        0.3 * vader_results["Negative"]),
            "Neutral": (0.4 * final_sentiment["Neutral"] + 
                       0.3 * normalized_zeroshot["Neutral"] + 
                       0.3 * vader_results["Neutral"]),
            "Positive": (0.4 * final_sentiment["Positive"] + 
                       0.3 * normalized_zeroshot["Positive"] + 
                       0.3 * vader_results["Positive"])
        }
        
        # Normalize to ensure sum to 1
        total = sum(ensemble_scores.values())
        ensemble_scores = {k: v/total for k, v in ensemble_scores.items()}
        
        # Determine final sentiment
        final_sentiment_label = max(ensemble_scores.items(), key=lambda x: x[1])[0]
        
        # If sarcasm is detected with high confidence, append to label
        if sarcasm_prob > 0.65:
            final_sentiment_label = f"{final_sentiment_label} (Sarcastic)"
        
        return {
            "Transformer": transformer_results,
            "ZeroShot": normalized_zeroshot,
            "VADER": vader_results,
            "Sarcasm": sarcasm_results,
            "Emotions": emotion_results,
            "Ensemble": ensemble_scores,
            "Final_Sentiment": final_sentiment_label
        }

def visualize_results(results: Dict) -> Tuple[str, Dict]:
    """Create visual representation of the analysis results"""
    final_sentiment = results["Final_Sentiment"]
    ensemble_scores = results["Ensemble"]
    
    # Create sentiment emoji
    sentiment_emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐",
        "Positive (Sarcastic)": "😏",
        "Negative (Sarcastic)": "🙄",
        "Neutral (Sarcastic)": "😒"
    }.get(final_sentiment, "🤔")
    
    # Format emotion results
    top_emotions = "\n".join(
        f"- {emo.capitalize()}: {score:.2%}" 
        for emo, score in results["Emotions"].items()
    )
    
    # Create markdown output
    markdown_output = f"""
    ## Enhanced Sentiment Analysis Results
    
    **Final Sentiment**: {final_sentiment} {sentiment_emoji}
    
    ### Confidence Scores:
    - Positive: {ensemble_scores['Positive']:.2%}
    - Neutral: {ensemble_scores['Neutral']:.2%}
    - Negative: {ensemble_scores['Negative']:.2%}
    
    ### Emotional Tone:
    {top_emotions}
    
    ### Sarcasm Detection:
    - Sarcastic: {results['Sarcasm'].get('Sarcastic', 0):.2%}
    - Not Sarcastic: {results['Sarcasm'].get('Not Sarcastic', 0):.2%}
    
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
analyzer = EnhancedSentimentAnalyzer()

def analyze_sentiment(text: str) -> Dict:
    """Main function to analyze sentiment of input text"""
    if not text.strip():
        return {"error": "Please enter some text to analyze."}
    
    results = analyzer.ensemble_analysis(text)
    if "error" in results:
        return results
    
    markdown_output, scores = visualize_results(results)
    
    return {
        "markdown": markdown_output,
        "scores": scores,
        "sentiment": results["Final_Sentiment"],
        "emotions": results["Emotions"],
        "sarcasm": results["Sarcasm"].get("Sarcastic", 0)
    }

# Create Gradio interface
with gr.Blocks(title="Enhanced Sentiment Analysis Tool", theme="soft") as demo:
    gr.Markdown("# 🌟 Enhanced AI Sentiment Analysis Tool")
    gr.Markdown("Analyze text sentiment with sarcasm and emotion detection")
    
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
                emotion_chart = gr.Label(label="Top Emotions")
                sarcasm_gauge = gr.Label(label="Sarcasm Probability")
    
    # Examples with sarcastic and complex emotional texts
    gr.Examples(
        examples=[
            ["I absolutely love this product! It's amazing and works perfectly."],
            ["The service was terrible and the staff was rude. Never coming back!"],
            ["This university is so great, I can't wait to fail all my exams here!"],
            ["Oh wonderful, another meeting that could have been an email."],
            ["I'm thrilled to be working late again tonight. Really, just overjoyed."],
            ["The movie was... interesting. If by interesting you mean boring."]
        ],
        inputs=input_text,
        label="Try these examples (including sarcastic ones)"
    )
    
    # Analysis function
    def analyze_and_display(text):
        result = analyze_sentiment(text)
        if "error" in result:
            return {
                output_markdown: f"## Error\n{result['error']}",
                sentiment_output: "Error",
                sentiment_scores: {},
                emotion_chart: {},
                sarcasm_gauge: {}
            }
        
        # Format emotions for display
        emotions_formatted = "\n".join(
            f"{emo.capitalize()}: {score:.2%}" 
            for emo, score in result["emotions"].items()
        )
        
        # Format sarcasm probability
        sarcasm_formatted = f"Sarcastic: {result['sarcasm']:.2%}"
        
        return {
            output_markdown: result["markdown"],
            sentiment_output: result["sentiment"],
            sentiment_scores: result["scores"],
            emotion_chart: emotions_formatted,
            sarcasm_gauge: sarcasm_formatted
        }
    
    analyze_btn.click(
        fn=analyze_and_display,
        inputs=input_text,
        outputs=[output_markdown, sentiment_output, sentiment_scores, emotion_chart, sarcasm_gauge]
    )

if __name__ == "__main__":
    # Run the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=7860)
