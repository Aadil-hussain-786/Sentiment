import gradio as gr
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer
)
from typing import Dict, Tuple, List
import numpy as np
from time import sleep
import warnings
from langdetect import detect

warnings.filterwarnings("ignore")

# Configuration
MODELS = {
    "en": {
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "emotion": "SamLowe/roberta-base-go_emotions",
        "sarcasm": "yiyanghkust/finbert-tone"
    },
    "multilingual": {
        "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
        "translation": "Helsinki-NLP/opus-mt-mul-en"
    }
}

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
models = {}
tokenizers = {}

print("Loading models...")
for lang, model_dict in MODELS.items():
    if lang == "multilingual":
        # Translation model for multilingual support
        translation_model_name = model_dict["translation"]
        models["translation"] = MarianMTModel.from_pretrained(translation_model_name).to(device)
        tokenizers["translation"] = MarianTokenizer.from_pretrained(translation_model_name)
        
        # Multilingual sentiment model
        sentiment_model_name = model_dict["sentiment"]
        models["multilingual_sentiment"] = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
        tokenizers["multilingual_sentiment"] = AutoTokenizer.from_pretrained(sentiment_model_name)
    else:
        for task, model_name in model_dict.items():
            key = f"{lang}_{task}"
            models[key] = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            tokenizers[key] = AutoTokenizer.from_pretrained(model_name)

print("Models loaded successfully!")

# Emotion labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# Translation functions
def translate_to_english(text: str, src_lang: str = None) -> str:
    """Translate text to English for analysis"""
    if src_lang and src_lang != "en":
        try:
            text = f">{src_lang} {text}"
            inputs = tokenizers["translation"](text, return_tensors="pt", truncation=True).to(device)
            translated = models["translation"].generate(**inputs)
            return tokenizers["translation"].decode(translated[0], skip_special_tokens=True)
        except:
            return text  # Fallback to original text if translation fails
    return text

# Sentiment analysis functions
def analyze_sentiment(text: str, lang: str = "en") -> Dict:
    """Analyze sentiment with detailed scores"""
    try:
        if lang != "en":
            # Use multilingual model
            inputs = tokenizers["multilingual_sentiment"](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(device)
            with torch.no_grad():
                outputs = models["multilingual_sentiment"](**inputs)
            
            scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            star_rating = np.argmax(scores) + 1  # 1-5 star rating
            
            # Convert star rating to sentiment
            if star_rating <= 2:
                sentiment = "negative"
            elif star_rating == 3:
                sentiment = "neutral"
            else:
                sentiment = "positive"
                
            return {
                "sentiment": sentiment,
                "scores": {
                    "negative": float(scores[0] + scores[1]) / 2,
                    "neutral": float(scores[2]),
                    "positive": float(scores[3] + scores[4]) / 2
                }
            }
        else:
            # Use English-specific model
            inputs = tokenizers["en_sentiment"](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(device)
            with torch.no_grad():
                outputs = models["en_sentiment"](**inputs)
            
            scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            sentiment_labels = ["negative", "neutral", "positive"]
            sentiment = sentiment_labels[np.argmax(scores)]
            
            return {
                "sentiment": sentiment,
                "scores": {
                    "negative": float(scores[0]),
                    "neutral": float(scores[1]),
                    "positive": float(scores[2])
                }
            }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {
            "sentiment": "neutral",
            "scores": {
                "negative": 0.0,
                "neutral": 1.0,
                "positive": 0.0
            }
        }

def detect_sarcasm(text: str) -> Dict:
    """Detect sarcasm with confidence score"""
    try:
        inputs = tokenizers["en_sarcasm"](
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = models["en_sarcasm"](**inputs)
        
        scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        sarcasm_prob = float(scores[1])  # Assuming index 1 is sarcasm
        
        return {
            "is_sarcastic": sarcasm_prob > 0.5,
            "confidence": sarcasm_prob
        }
    except Exception as e:
        print(f"Sarcasm detection error: {e}")
        return {
            "is_sarcastic": False,
            "confidence": 0.0
        }

def analyze_emotion(text: str) -> Dict:
    """Analyze emotional content"""
    try:
        inputs = tokenizers["en_emotion"](
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = models["en_emotion"](**inputs)
        
        scores = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        top_emotions = sorted(zip(EMOTION_LABELS, scores), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "emotions": {emotion: float(score) for emotion, score in top_emotions},
            "dominant_emotion": top_emotions[0][0]
        }
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return {
            "emotions": {"neutral": 1.0},
            "dominant_emotion": "neutral"
        }

def detect_language(text: str) -> str:
    """Language detection using langdetect"""
    try:
        lang = detect(text)
        return lang if lang in ['en', 'es', 'fr', 'de'] else 'en'
    except:
        return 'en'

def comprehensive_analysis(text: str) -> Tuple[Dict, str]:
    """Perform comprehensive sentiment, emotion, and sarcasm analysis"""
    try:
        # Detect language
        lang = detect_language(text)
        translated_text = text if lang == "en" else translate_to_english(text, lang)
        
        # Get sentiment
        sentiment_result = analyze_sentiment(text, lang)
        
        # Get sarcasm (English only)
        sarcasm_result = detect_sarcasm(translated_text) if lang == "en" else {"is_sarcastic": False, "confidence": 0.0}
        
        # Get emotions (English only)
        emotion_result = analyze_emotion(translated_text) if lang == "en" else {"emotions": {}, "dominant_emotion": "neutral"}
        
        # Adjust sentiment if sarcasm is detected
        final_sentiment = sentiment_result["sentiment"]
        if sarcasm_result["is_sarcastic"] and sarcasm_result["confidence"] > 0.6:
            if final_sentiment == "positive":
                final_sentiment = "negative"
            elif final_sentiment == "negative":
                final_sentiment = "positive"
        
        # Prepare detailed results
        detailed_results = {
            "text": text,
            "language": lang,
            "sentiment": final_sentiment,
            "sentiment_scores": sentiment_result["scores"],
            "sarcasm_detected": sarcasm_result["is_sarcastic"],
            "sarcasm_confidence": sarcasm_result["confidence"],
            "emotions": emotion_result["emotions"],
            "dominant_emotion": emotion_result["dominant_emotion"],
            "translated_text": translated_text if lang != "en" else None
        }
        
        # Prepare human-readable response
        response_parts = []
        
        # Apply sentiment color
        sentiment_display = f"Sentiment: {final_sentiment.upper()}"
        if final_sentiment == "positive":
            sentiment_display = f"<span class='sentiment-positive'>{sentiment_display}</span>"
        elif final_sentiment == "negative":
            sentiment_display = f"<span class='sentiment-negative'>{sentiment_display}</span>"
        else:
            sentiment_display = f"<span class='sentiment-neutral'>{sentiment_display}</span>"
        
        response_parts.append(sentiment_display)
        
        if lang != "en":
            response_parts.append(f"Detected Language: {lang.upper()}")
        
        if sarcasm_result["is_sarcastic"]:
            sarcasm_display = f"⚠️ <span class='sarcasm-detected'>Sarcasm detected: {sarcasm_result['confidence']*100:.1f}% confidence</span>"
            response_parts.append(sarcasm_display)
        
        if emotion_result["emotions"]:
            top_emotion, top_score = next(iter(emotion_result["emotions"].items()))
            response_parts.append(f"Dominant Emotion: {top_emotion.capitalize()} ({top_score*100:.1f}%)")
        
        if sentiment_result["sentiment"] != final_sentiment and sarcasm_result["is_sarcastic"]:
            response_parts.append(f"Note: Sentiment reversed due to sarcasm detection")
        
        response = "<br>".join(response_parts)
        
        return detailed_results, response
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return {
            "text": text,
            "language": "en",
            "sentiment": "neutral",
            "sentiment_scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            "sarcasm_detected": False,
            "sarcasm_confidence": 0.0,
            "emotions": {"neutral": 1.0},
            "dominant_emotion": "neutral",
            "translated_text": text
        }, "Error in analysis. Please try again."

# Gradio UI with typing animation
def respond(message: str, history: List[List[str]]):
    """Chatbot response function with typing animation"""
    # First yield the message with user avatar
    history.append([message, None])
    yield history
    
    # Simulate typing animation
    for i in range(3):
        history[-1][1] = "..." * (i + 1)
        yield history
        sleep(0.1)
    
    # Process the message
    analysis, response = comprehensive_analysis(message)
    
    # Simulate typing out the response
    typed_response = ""
    for char in response:
        typed_response += char
        history[-1][1] = typed_response
        sleep(0.02)
        yield history

# Custom CSS for advanced UI
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chatbot {
    min-height: 400px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.dark .chatbot {
    background-color: #1f1f1f;
}
.textbox textarea {
    font-size: 16px;
    padding: 12px;
    border-radius: 8px;
}
.sentiment-positive {
    color: #10B981;
    font-weight: bold;
}
.sentiment-negative {
    color: #EF4444;
    font-weight: bold;
}
.sentiment-neutral {
    color: #6B7280;
    font-weight: bold;
}
.sarcasm-detected {
    color: #F59E0B;
    font-weight: bold;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Advanced Sentiment Analysis Chatbot")
    gr.Markdown("This tool analyzes sentiment, emotion, and sarcasm in multiple languages with high accuracy.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                avatar_images=(
                    "https://i.imgur.com/7aXU3dy.png",  # User avatar
                    "https://i.imgur.com/qh3UygJ.png"   # Bot avatar
                ),
                render_markdown=True
            )
            msg = gr.Textbox(
                placeholder="Type a message in any language...",
                label="Message",
                autofocus=True
            )
            btn = gr.Button("Analyze")
            clear = gr.ClearButton([msg, chatbot])
        
        with gr.Column(scale=1):
            gr.Markdown("## Analysis Details")
            sentiment_output = gr.Label(label="Sentiment")
            emotion_output = gr.Label(label="Top Emotions")
            sarcasm_output = gr.Label(label="Sarcasm Detection")
            language_output = gr.Label(label="Detected Language")
            raw_output = gr.JSON(label="Raw Analysis Data")
    
    # Event handlers
    def process_message(message: str):
        analysis, _ = comprehensive_analysis(message)
        return (
            {"label": analysis["sentiment"].upper()},
            {"label": ", ".join([f"{k} ({v:.2f})" for k, v in analysis["emotions"].items()])},
            {"label": f"{analysis['sarcasm_confidence']*100:.1f}% confident" if analysis["sarcasm_detected"] else "Not detected"},
            {"label": analysis["language"].upper()},
            analysis
        )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        fn=process_message,
        inputs=msg,
        outputs=[
            sentiment_output,
            emotion_output,
            sarcasm_output,
            language_output,
            raw_output
        ]
    )
    
    btn.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        fn=process_message,
        inputs=msg,
        outputs=[
            sentiment_output,
            emotion_output,
            sarcasm_output,
            language_output,
            raw_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
