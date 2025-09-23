#!/usr/bin/env python3
"""
Test script for the trained MindTrack model
"""

import sys
import os
sys.path.append('.')

from app.utils import load_model_and_tokenizer, predict_single_text

def test_model():
    """Test the trained model with various inputs"""
    
    print("Loading trained model...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded successfully!")
    
    # Test cases
    test_cases = [
        "I have been feeling really depressed lately and sometimes think about ending it all",
        "I am so happy today, life is wonderful and full of possibilities",
        "I don't see the point in living anymore, everything is hopeless",
        "Just had an amazing day with friends, feeling grateful for life",
        "I feel worthless and like nobody would miss me if I was gone"
    ]
    
    print("\n" + "="*80)
    print("TESTING RETRAINED MODEL")
    print("="*80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        prediction_label, confidence, probabilities = predict_single_text(text)
        
        print(f"Prediction: {prediction_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Risk probability: {probabilities[1]:.4f}")
        print(f"Normal probability: {probabilities[0]:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    test_model()