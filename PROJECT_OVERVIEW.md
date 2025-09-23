# MindTrack Project Structure Overview

## ✅ Complete Project Setup

Your MindTrack application has been successfully created with the following structure:

```
MindTrack/
├── .streamlit/                    # Streamlit configuration
│   ├── config.toml               # App configuration
│   └── secrets_template.toml     # Template for Reddit API credentials
├── app/                          # Main application code
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Streamlit web application (MAIN FILE)
│   └── utils.py                 # Helper functions for ML and LIME
├── data/                         # Dataset storage
│   └── mental_health_data.csv   # Sample dataset (50 labeled examples)
├── model/                        # ML model code
│   ├── __init__.py              # Package initialization
│   ├── train.py                 # Model training script
│   └── saved_model/             # Trained model storage (created after training)
├── venv/                        # Virtual environment (ACTIVE)
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Python dependencies (INSTALLED ✅)
├── setup.py                     # Setup automation script
├── test_setup.py               # Test validation script (ALL TESTS PASS ✅)
├── run_app.ps1                 # PowerShell launcher
└── run_app.bat                 # Batch file launcher
```

## 🎯 Features Implemented

### 1. Single Post Analyzer
- ✅ Text input interface
- ✅ AI-powered sentiment analysis
- ✅ Confidence scoring
- ✅ LIME explainability (highlights important words)
- ✅ Probability breakdown visualization

### 2. Live Subreddit Monitor
- ✅ Real-time Reddit post streaming
- ✅ Automatic sentiment analysis
- ✅ Live dashboard updates
- ✅ Reddit API integration (PRAW)
- ✅ Configurable subreddit monitoring

### 3. AI/ML Components
- ✅ DistilBERT transformer model
- ✅ Fallback model (works without training)
- ✅ Custom training pipeline
- ✅ Model evaluation metrics
- ✅ Automatic model loading

### 4. Explainability
- ✅ LIME text explanations
- ✅ Word importance highlighting
- ✅ Feature importance scores
- ✅ Interactive explanations

## 🚀 How to Run

### Option 1: PowerShell (Recommended)
```powershell
.\run_app.ps1
```

### Option 2: Command Line
```cmd
run_app.bat
```

### Option 3: Direct Command
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run the app
python -m streamlit run app/app.py
```

## 🔧 Configuration

### Reddit API Setup (Optional - for Live Monitor)
1. Go to https://www.reddit.com/prefs/apps
2. Create a new application (script type)
3. Copy `.streamlit/secrets_template.toml` to `.streamlit/secrets.toml`
4. Add your credentials:
```toml
[reddit]
client_id = "your_client_id"
client_secret = "your_client_secret"
user_agent = "MindTrack:v1.0 (by /u/yourusername)"
```

## 📊 Sample Usage

### Single Post Analysis
1. Navigate to "Single Post Analyzer"
2. Enter text: "I'm feeling really down and hopeless"
3. Click "Analyze Post"
4. View prediction, confidence, and LIME explanation

### Live Monitoring
1. Navigate to "Live Subreddit Monitor"
2. Enter subreddit: "mentalhealthsupport"
3. Add Reddit API credentials
4. Click "Start Monitoring"
5. Watch real-time analysis

## 🎓 Model Training (Optional)

To train your own model:
```bash
cd model
python train.py
```

The app works with the fallback model, but training improves accuracy.

## ✅ Validation Results

All tests passed successfully:
- ✅ All dependencies installed
- ✅ Model loading functional
- ✅ Prediction working
- ✅ Sample data loaded
- ✅ Virtual environment active

## 🔒 Important Notes

⚠️ **Disclaimer**: This tool is for educational purposes only. It should not be used as a substitute for professional mental health advice.

🛡️ **Privacy**: No data is stored or transmitted externally. All processing happens locally.

📈 **Performance**: The fallback model provides basic functionality. Train the model for better accuracy.

## 🎉 You're Ready!

Your MindTrack application is fully set up and ready to use. Simply run one of the launch commands above and start analyzing mental health sentiment in text!