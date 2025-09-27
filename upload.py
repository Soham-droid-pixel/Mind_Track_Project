#!/usr/bin/env python3
"""
MindTrack Model Upload Script
Uploads the trained model to Hugging Face Hub for deployment.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch

# Configuration - Use your actual HF username
HF_USERNAME = "techhy"  # Your actual HF username from whoami command
REPO_NAME = "mindtrack-mental-health-analyzer"
MODEL_PATH = "./saved_model"

def verify_model_files():
    """Verify that all required model files exist."""
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Error: Model directory '{MODEL_PATH}' not found!")
        return False
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        return False
    
    print("✅ All required model files found!")
    return True

def load_training_metadata():
    """Load training metadata if available."""
    metadata_path = Path(MODEL_PATH) / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def create_model_card():
    """Create a comprehensive model card for Hugging Face Hub."""
    metadata = load_training_metadata()
    
    accuracy = metadata.get('accuracy', 'N/A')
    f1_score = metadata.get('f1_score', 'N/A')
    
    model_card = f"""---
language: en
license: mit
tags:
- mental-health
- sentiment-analysis
- pytorch
- transformers
- distilbert
datasets:
- custom-mental-health-dataset
pipeline_tag: text-classification
widget:
- text: "I am feeling great today and everything is going well!"
  example_title: "Positive Mental State"
- text: "I don't see the point in anything anymore, everything feels hopeless"
  example_title: "Mental Health Risk"
- text: "Just having a regular day at work, nothing special"
  example_title: "Neutral State"
model-index:
- name: MindTrack Mental Health Analyzer
  results:
  - task:
      type: text-classification
      name: Mental Health Risk Detection
    metrics:
    - type: accuracy
      value: {accuracy}
      name: Accuracy
    - type: f1
      value: {f1_score}
      name: F1 Score
---

# MindTrack: Mental Health Sentiment Analyzer 🧠

## Model Description

MindTrack is a fine-tuned DistilBERT model specifically trained for mental health sentiment analysis and risk detection in text content. The model can classify text into two categories:

- **Normal**: Indicates healthy mental state or neutral content
- **Risk Detected**: Indicates potential mental health concerns that may require attention

## Model Details

- **Model Type**: DistilBERT for Sequence Classification
- **Training Data**: Curated mental health dataset with balanced samples
- **Languages**: English
- **License**: MIT

## Performance

- **Accuracy**: {accuracy}
- **F1 Score**: {f1_score}
- **Training Samples**: {metadata.get('num_samples', 'N/A')}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_id = "{HF_USERNAME}/{REPO_NAME}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Example usage
text = "I am feeling great today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

labels = {{0: "Normal", 1: "Risk Detected"}}
print(f"Prediction: {{labels[predicted_class]}} (Confidence: {{confidence:.2%}})")
```

## Intended Use

This model is designed to assist in identifying potential mental health concerns in text content. **Important**: This tool should not replace professional medical advice or diagnosis. Always consult qualified healthcare professionals for mental health issues.

## Limitations

- Trained primarily on English text
- May not capture cultural nuances in mental health expression
- Performance may vary on text significantly different from training data
- Should be used as a screening tool, not for final diagnosis

## Training Data

The model was trained on a carefully curated dataset of mental health-related text, including:
- Social media posts (anonymized)
- Mental health support forum discussions
- Clinical text samples (anonymized)
- Balanced representation of risk and normal states

## Ethical Considerations

- **Privacy**: No personal information was used in training
- **Bias**: Efforts were made to reduce bias, but some may remain
- **Responsible Use**: Should be used to help people, not to discriminate
- **Professional Guidance**: Always recommend professional help for mental health concerns

## Citation

If you use this model in your research or applications, please cite:

```
@misc{{mindtrack2024,
  title={{MindTrack: Mental Health Sentiment Analyzer}},
  author={{Soham}},
  year={{2024}},
  url={{https://huggingface.co/{HF_USERNAME}/{REPO_NAME}}}
}}
```

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Soham-droid-pixel/Mind_Track_Project).
"""
    
    return model_card

def upload_model():
    """Upload the model to Hugging Face Hub."""
    print(f"🚀 Starting upload to Hugging Face Hub...")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"🎯 Repository: {HF_USERNAME}/{REPO_NAME}")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository if it doesn't exist
        print("📝 Creating repository...")
        try:
            repo_url = create_repo(
                repo_id=f"{HF_USERNAME}/{REPO_NAME}",
                exist_ok=True,
                private=False
            )
            print(f"✅ Repository ready: {repo_url}")
        except Exception as e:
            print(f"⚠️  Repository may already exist: {e}")
        
        # Load and verify model
        print("🔍 Loading and verifying model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Test model with a sample prediction
        test_text = "I am feeling great today!"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"✅ Model test successful - output shape: {outputs.logits.shape}")
        
        # Upload model files
        print("⬆️  Uploading model to Hugging Face Hub...")
        model.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}")
        tokenizer.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}")
        
        # Create and upload model card
        print("📄 Creating model card...")
        model_card = create_model_card()
        
        # Save model card locally then upload
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(model_card)
        
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            commit_message="Add comprehensive model card"
        )
        
        # Clean up local README
        os.remove("README.md")
        
        print("✅ Upload completed successfully!")
        print(f"🌟 Your model is now available at: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
        print(f"🔗 Model ID for your app: {HF_USERNAME}/{REPO_NAME}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        print("💡 Make sure you're logged in: huggingface-cli login")
        return False

def main():
    """Main execution function."""
    print("🧠 MindTrack Model Upload Script")
    print("=" * 50)
    
    # Verify model files exist
    if not verify_model_files():
        sys.exit(1)
    
    # Upload model
    success = upload_model()
    
    if success:
        print("\n🎉 SUCCESS! Your model is now hosted on Hugging Face Hub!")
        print(f"📝 Next steps:")
        print(f"   1. Update app/utils.py to use: {HF_USERNAME}/{REPO_NAME}")
        print(f"   2. Remove saved_model/ directory from Git")
        print(f"   3. Commit and push to GitHub")
        print(f"   4. Deploy to Streamlit Cloud")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()