# MindTrack: AI-Powered Mental Health Sentiment Analyzer 🧠---

language: en

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mindtrack-analyzer.streamlit.app)license: mit

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/techhy/mindtrack-mental-health-analyzer)tags:

- mental-health

MindTrack is an advanced AI application that analyzes text content for mental health risk indicators using state-of-the-art natural language processing. Built with fine-tuned DistilBERT and enhanced with explainable AI features.- sentiment-analysis

- pytorch

## ✨ Features- transformers

- distilbert

- **🤖 Advanced AI Model**: Fine-tuned DistilBERT achieving 97%+ accuracydatasets:

- **🔍 Explainable AI**: LIME explanations show which words influenced predictions- custom-mental-health-dataset

- **📱 Reddit Integration**: Real-time monitoring of mental health subredditspipeline_tag: text-classification

- **⚡ Fast Performance**: Optimized for quick analysis and deploymentwidget:

- **🌐 Web Interface**: Beautiful Streamlit dashboard with modern UI- text: "I am feeling great today and everything is going well!"

- **☁️ Cloud-Ready**: Deployed model on Hugging Face Hub for scalability  example_title: "Positive Mental State"

- text: "I don't see the point in anything anymore, everything feels hopeless"

## 🚀 Quick Start  example_title: "Mental Health Risk"

- text: "Just having a regular day at work, nothing special"

### Online Demo  example_title: "Neutral State"

Visit the live application: [MindTrack on Streamlit Cloud](https://mindtrack-analyzer.streamlit.app)model-index:

- name: MindTrack Mental Health Analyzer

### Local Installation  results:

  - task:

1. **Clone the repository**      type: text-classification

   ```bash      name: Mental Health Risk Detection

   git clone https://github.com/Soham-droid-pixel/Mind_Track_Project.git    metrics:

   cd Mind_Track_Project    - type: accuracy

   ```      value: N/A

      name: Accuracy

2. **Install dependencies**    - type: f1

   ```bash      value: N/A

   pip install -r requirements.txt      name: F1 Score

   ```---



3. **Run the application**# MindTrack: Mental Health Sentiment Analyzer 🧠

   ```bash

   streamlit run app/app.py## Model Description

   ```

MindTrack is a fine-tuned DistilBERT model specifically trained for mental health sentiment analysis and risk detection in text content. The model can classify text into two categories:

## 🎯 How It Works

- **Normal**: Indicates healthy mental state or neutral content

MindTrack uses a fine-tuned DistilBERT model hosted on Hugging Face Hub to classify text into two categories:- **Risk Detected**: Indicates potential mental health concerns that may require attention



- **Normal**: Healthy mental state or neutral content## Model Details

- **Risk Detected**: Potential mental health concerns requiring attention

- **Model Type**: DistilBERT for Sequence Classification

### Model Performance- **Training Data**: Curated mental health dataset with balanced samples

- **Accuracy**: 97.13%- **Languages**: English

- **F1-Score**: 97.16% (Normal), 97.10% (Risk)- **License**: MIT

- **Confidence**: Provides reliable confidence scores for predictions

## Performance

## 🛠️ Technology Stack

- **Accuracy**: N/A

- **Frontend**: Streamlit with custom CSS styling- **F1 Score**: N/A

- **ML Model**: Fine-tuned DistilBERT (hosted on Hugging Face Hub)- **Training Samples**: N/A

- **Explainability**: LIME (Local Interpretable Model-agnostic Explanations)

- **APIs**: Reddit API (PRAW) for social media monitoring## Usage

- **Deployment**: Streamlit Community Cloud

```python

## 📊 Use Casesfrom transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch

1. **Content Moderation**: Identify potentially harmful content

2. **Mental Health Screening**: Early detection of concerning language patterns# Load model and tokenizer

3. **Social Media Monitoring**: Track mental health trends in communitiesmodel_id = "techhy/mindtrack-mental-health-analyzer"

4. **Research Tool**: Analyze large datasets for mental health indicatorstokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(model_id)

## ⚠️ Important Disclaimer

# Example usage

This tool is designed to assist in identifying potential mental health concerns but should **never replace professional medical advice**. If you or someone you know is experiencing mental health difficulties, please seek help from qualified professionals.text = "I am feeling great today!"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

### Crisis Resources

- **National Suicide Prevention Lifeline**: 988with torch.no_grad():

- **Crisis Text Line**: Text HOME to 741741    outputs = model(**inputs)

- **International Association for Suicide Prevention**: [https://www.iasp.info](https://www.iasp.info)    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()

## 🏗️ Architecture    confidence = probabilities[0][predicted_class].item()



```labels = {0: "Normal", 1: "Risk Detected"}

MindTrack/print(f"Prediction: {labels[predicted_class]} (Confidence: {confidence:.2%})")

├── app/```

│   ├── app.py          # Main Streamlit application

│   ├── utils.py        # Model loading and prediction utilities## Intended Use

├── data/               # Training data (not included for privacy)

├── model/              # Training scriptsThis model is designed to assist in identifying potential mental health concerns in text content. **Important**: This tool should not replace professional medical advice or diagnosis. Always consult qualified healthcare professionals for mental health issues.

├── requirements.txt    # Python dependencies

└── upload.py          # Model upload script to HF Hub## Limitations

```

- Trained primarily on English text

## 📈 Model Details- May not capture cultural nuances in mental health expression

- Performance may vary on text significantly different from training data

The MindTrack model is available on Hugging Face Hub: [`techhy/mindtrack-mental-health-analyzer`](https://huggingface.co/techhy/mindtrack-mental-health-analyzer)- Should be used as a screening tool, not for final diagnosis



### Usage Example## Training Data

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassificationThe model was trained on a carefully curated dataset of mental health-related text, including:

import torch- Social media posts (anonymized)

- Mental health support forum discussions

model_id = "techhy/mindtrack-mental-health-analyzer"- Clinical text samples (anonymized)

tokenizer = AutoTokenizer.from_pretrained(model_id)- Balanced representation of risk and normal states

model = AutoModelForSequenceClassification.from_pretrained(model_id)

## Ethical Considerations

text = "I am feeling great today!"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)- **Privacy**: No personal information was used in training

- **Bias**: Efforts were made to reduce bias, but some may remain

with torch.no_grad():- **Responsible Use**: Should be used to help people, not to discriminate

    outputs = model(**inputs)- **Professional Guidance**: Always recommend professional help for mental health concerns

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()## Citation



labels = {0: "Normal", 1: "Risk Detected"}If you use this model in your research or applications, please cite:

print(f"Prediction: {labels[predicted_class]}")

``````

@misc{mindtrack2024,

## 🤝 Contributing  title={MindTrack: Mental Health Sentiment Analyzer},

  author={Soham},

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.  year={2024},

  url={https://huggingface.co/techhy/mindtrack-mental-health-analyzer}

## 📄 License}

```

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

## 🙏 Acknowledgments

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Soham-droid-pixel/Mind_Track_Project).

- **Hugging Face** for the transformers library and model hosting
- **Streamlit** for the amazing web framework
- **LIME** for explainable AI capabilities
- **Reddit API** for social media integration

## 📞 Contact

**Soham** - [@Soham-droid-pixel](https://github.com/Soham-droid-pixel)

Project Link: [https://github.com/Soham-droid-pixel/Mind_Track_Project](https://github.com/Soham-droid-pixel/Mind_Track_Project)

---

⭐ **If you find this project helpful, please consider giving it a star!**