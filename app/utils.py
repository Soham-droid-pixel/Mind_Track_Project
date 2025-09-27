"""
MindTrack Utility Functions
This module contains helper functions for model loading, prediction, and explanation.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - Hugging Face Hub
MODEL_ID = "techhy/mindtrack-mental-health-analyzer"  # Hosted model on HF Hub

# Global variables to store model and tokenizer
_model = None
_tokenizer = None

def load_model_and_tokenizer(model_id=None):
    """
    Load the model and tokenizer from Hugging Face Hub.
    
    Args:
        model_id (str): Hugging Face model ID (uses default if None)
        
    Returns:
        tuple: Loaded model and tokenizer
    """
    global _model, _tokenizer
    
    # Use provided model_id or default
    if model_id is None:
        model_id = MODEL_ID
    
    try:
        logger.info(f"Loading model from Hugging Face Hub: {model_id}")
        
        # Load tokenizer and model from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        # Set model to evaluation mode
        model.eval()
        
        # Store globally for reuse
        _model = model
        _tokenizer = tokenizer
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return load_fallback_model()

def load_fallback_model():
    """
    Load a fallback pre-trained model if the trained model is not available.
    
    Returns:
        tuple: Fallback model and tokenizer
    """
    logger.info("Loading fallback DistilBERT model...")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NORMAL", 1: "RISK"},
        label2id={"NORMAL": 0, "RISK": 1}
    )
    model.eval()
    
    return model, tokenizer

def predictor(texts):
    """
    Prediction function for LIME explainer.
    Takes a list of raw text strings and returns prediction probabilities.
    
    Args:
        texts (list): List of text strings to predict
        
    Returns:
        numpy.ndarray: Array of shape (num_texts, num_classes) with probabilities
    """
    global _model, _tokenizer
    
    # Ensure model and tokenizer are loaded
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model_and_tokenizer()
    
    # Tokenize the texts
    inputs = _tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
    return probabilities.numpy()

def predict_single_text(text, model=None, tokenizer=None):
    """
    Predict mental health risk for a single text.
    
    Args:
        text (str): Input text to analyze
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        
    Returns:
        tuple: (prediction_label, confidence_score, probabilities)
    """
    global _model, _tokenizer
    
    # Use provided model/tokenizer or load global ones
    if model is None or tokenizer is None:
        if _model is None or _tokenizer is None:
            _model, _tokenizer = load_model_and_tokenizer()
        model, tokenizer = _model, _tokenizer
    
    # Get prediction probabilities
    probabilities = predictor([text])[0]
    
    # Get prediction label and confidence
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    # Map class to label
    labels = {0: "Normal", 1: "Risk Detected"}
    prediction_label = labels[predicted_class]
    
    return prediction_label, confidence, probabilities

def explain_text(explainer, text, predictor_fn, num_features=10):
    """
    Generate LIME explanation for a text prediction.
    
    Args:
        explainer: LimeTextExplainer instance
        text (str): Text to explain
        predictor_fn: Function that returns prediction probabilities
        num_features (int): Number of features to include in explanation
        
    Returns:
        explanation: LIME explanation object
    """
    try:
        # Generate explanation
        explanation = explainer.explain_instance(
            text,
            predictor_fn,
            num_features=num_features,
            labels=[0, 1]  # Explain both classes
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return None

def format_lime_explanation(explanation, prediction_label):
    """
    Format LIME explanation into user-friendly format.
    
    Args:
        explanation: LIME explanation object
        prediction_label: The predicted label
        
    Returns:
        dict: Formatted explanation with positive and negative influences
    """
    if explanation is None:
        return None
    
    # Get the explanation for the predicted class
    class_idx = 1 if prediction_label == "Risk Detected" else 0
    exp_list = explanation.as_list(label=class_idx)
    
    # Separate positive and negative influences
    positive_words = []
    negative_words = []
    
    for word, importance in exp_list:
        if importance > 0:
            positive_words.append({
                'word': word,
                'importance': importance,
                'description': f"'{word}' increases {prediction_label.lower()} prediction"
            })
        else:
            negative_words.append({
                'word': word,
                'importance': abs(importance),
                'description': f"'{word}' decreases {prediction_label.lower()} prediction"
            })
    
    # Sort by importance
    positive_words.sort(key=lambda x: x['importance'], reverse=True)
    negative_words.sort(key=lambda x: x['importance'], reverse=True)
    
    return {
        'positive_influences': positive_words[:5],  # Top 5 positive influences
        'negative_influences': negative_words[:5],  # Top 5 negative influences
        'prediction': prediction_label,
        'explanation_summary': _get_explanation_summary(positive_words, negative_words, prediction_label)
    }

def _get_explanation_summary(positive_words, negative_words, prediction_label):
    """Generate a human-readable summary of the explanation."""
    if not positive_words and not negative_words:
        return "No significant word influences found."
    
    summary = []
    
    if prediction_label == "Risk Detected":
        if positive_words:
            top_risk_words = [w['word'] for w in positive_words[:3]]
            summary.append(f"🚨 The model detected risk primarily due to words like: {', '.join(top_risk_words)}")
        
        if negative_words:
            top_safe_words = [w['word'] for w in negative_words[:2]]
            summary.append(f"✅ However, words like '{', '.join(top_safe_words)}' suggest some positive elements.")
    else:
        if positive_words:
            top_normal_words = [w['word'] for w in positive_words[:3]]
            summary.append(f"✅ The model classified this as normal due to words like: {', '.join(top_normal_words)}")
        
        if negative_words:
            top_concern_words = [w['word'] for w in negative_words[:2]]
            summary.append(f"⚠️ Words like '{', '.join(top_concern_words)}' showed some concerning signals.")
    
    return " ".join(summary)

def create_lime_explainer():
    """
    Create and configure a LIME text explainer.
    
    Returns:
        LimeTextExplainer: Configured explainer instance
    """
    explainer = LimeTextExplainer(
        class_names=['Normal', 'Risk'],
        feature_selection='forward_selection',
        split_expression=' ',
        bow=True
    )
    
    return explainer

def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        dict: Model information
    """
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        return {"status": "No model loaded"}
    
    info = {
        "status": "Model loaded",
        "model_type": _model.__class__.__name__,
        "num_parameters": sum(p.numel() for p in _model.parameters()),
        "vocab_size": _tokenizer.vocab_size if hasattr(_tokenizer, 'vocab_size') else "Unknown"
    }
    
    return info

def preprocess_text(text):
    """
    Basic text preprocessing for better model performance.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

def batch_predict(texts, batch_size=32):
    """
    Predict mental health risk for a batch of texts.
    
    Args:
        texts (list): List of texts to analyze
        batch_size (int): Number of texts to process at once
        
    Returns:
        list: List of tuples (prediction_label, confidence, probabilities)
    """
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_probabilities = predictor(batch_texts)
        
        for j, probabilities in enumerate(batch_probabilities):
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            labels = {0: "Normal", 1: "Risk Detected"}
            prediction_label = labels[predicted_class]
            
            results.append((prediction_label, confidence, probabilities))
    
    return results