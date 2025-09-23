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

# Global variables to store model and tokenizer
_model = None
_tokenizer = None

def load_model_and_tokenizer(model_path="./model/saved_model"):
    """
    Load the saved model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the saved model directory
        
    Returns:
        tuple: Loaded model and tokenizer
    """
    global _model, _tokenizer
    
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist. Using fallback model.")
            return load_fallback_model()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
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