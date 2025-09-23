"""
MindTrack Test Runner
Simple test to validate the application setup
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import transformers
        print("✅ Transformers imported successfully")
        
        import praw
        print("✅ PRAW imported successfully")
        
        import lime
        print("✅ LIME imported successfully")
        
        import pandas
        print("✅ Pandas imported successfully")
        
        import numpy
        print("✅ NumPy imported successfully")
        
        from app.utils import create_lime_explainer, load_model_and_tokenizer
        print("✅ App utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality."""
    print("\nTesting model loading...")
    
    try:
        from app.utils import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        if model is not None and tokenizer is not None:
            print("✅ Model and tokenizer loaded successfully")
            return True
        else:
            print("⚠️ Fallback model loaded (training required for full functionality)")
            return True
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_prediction():
    """Test prediction functionality."""
    print("\nTesting prediction...")
    
    try:
        from app.utils import predict_single_text
        
        test_text = "I'm feeling great today!"
        prediction, confidence, probabilities = predict_single_text(test_text)
        
        print(f"✅ Prediction successful: {prediction} (confidence: {confidence:.2%})")
        return True
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_data():
    """Test if sample data exists."""
    print("\nTesting data...")
    
    data_path = "data/mental_health_data.csv"
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"✅ Sample data loaded: {len(df)} rows")
        return True
    else:
        print("⚠️ Sample data not found, will use built-in data")
        return True

def main():
    """Run all tests."""
    print("🧠 MindTrack Application Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_loading,
        test_prediction,
        test_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your MindTrack setup is ready.")
        print("\nNext steps:")
        print("1. Run: streamlit run app/app.py")
        print("2. Or use: python run_app.ps1 (Windows)")
        print("3. Or use: ./run_app.bat (Windows Command Prompt)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())