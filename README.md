# MindTrack: Mental Health Sentiment Analyzer

MindTrack is an NLP-based mental health sentiment analyzer that can analyze social media posts for mental health risk indicators. It features both single post analysis with LIME explanations and live subreddit monitoring capabilities.

## Features

1. **Single Post Analyzer**: Analyze individual text posts with AI-powered sentiment analysis
2. **Live Subreddit Monitor**: Real-time monitoring of Reddit posts from specified subreddits
3. **LIME Explanations**: Understand which words influenced the AI's decision
4. **Interactive Dashboard**: User-friendly Streamlit web interface

## Technology Stack

- Python 3.9+
- Streamlit for web dashboard
- PyTorch and Transformers (Hugging Face) for NLP
- PRAW for Reddit API integration
- LIME for model explainability
- Pandas for data handling

## Project Structure

```
MindTrack/
├── app/
│   ├── __init__.py
│   ├── app.py              # Main Streamlit application
│   └── utils.py            # Helper functions
├── model/
│   ├── __init__.py
│   ├── train.py            # Model training script
│   └── saved_model/        # Trained model files (created after training)
├── data/
│   └── mental_health_data.csv  # Sample dataset
├── .streamlit/
│   ├── config.toml         # Streamlit configuration
│   └── secrets_template.toml   # Template for API secrets
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation and Setup

### 1. Clone/Download the Project

Ensure you have the MindTrack folder with all the files.

### 2. Create Virtual Environment

```bash
cd MindTrack
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Reddit API (Optional - for Live Monitor)

1. Go to https://www.reddit.com/prefs/apps
2. Create a new application (script type)
3. Copy the client ID and secret
4. Copy `.streamlit/secrets_template.toml` to `.streamlit/secrets.toml`
5. Fill in your Reddit API credentials in `secrets.toml`

### 5. Train the Model (Optional)

```bash
cd model
python train.py
```

Note: The app includes a fallback model that will work even without training.

### 6. Run the Application

```bash
streamlit run app/app.py
```

The application will open in your browser at http://localhost:8501

## Usage

### Single Post Analyzer

1. Navigate to "Single Post Analyzer" in the sidebar
2. Enter text in the text area
3. Click "Analyze Post"
4. View the prediction, confidence score, and LIME explanation

### Live Subreddit Monitor

1. Navigate to "Live Subreddit Monitor" in the sidebar
2. Enter Reddit API credentials (if not in secrets.toml)
3. Specify the subreddit to monitor
4. Click "Start Monitoring"
5. Watch real-time analysis of new posts

## Model Training

The training script (`model/train.py`) includes:

- Data loading and preprocessing
- DistilBERT fine-tuning
- Model evaluation with accuracy and F1-score
- Model saving for deployment

To train with your own data:

1. Replace `data/mental_health_data.csv` with your dataset
2. Ensure columns are named 'text' and 'label' (0 = Normal, 1 = Risk)
3. Run `python model/train.py`

## Sample Data

The included dataset contains 50 sample text posts labeled for mental health risk:
- Label 0: Normal/Positive posts
- Label 1: Risk/Concerning posts

## API Integration

### Reddit API Setup

1. Create a Reddit account
2. Go to https://www.reddit.com/prefs/apps
3. Click "Create App" or "Create Another App"
4. Choose "script" as the app type
5. Note down the client ID (under the app name) and secret

### Configuration

Add your credentials to `.streamlit/secrets.toml`:

```toml
[reddit]
client_id = "your_client_id_here"
client_secret = "your_client_secret_here"
user_agent = "MindTrack:v1.0 (by /u/yourusername)"
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: The app will use a fallback model if the trained model isn't available
2. **Reddit API Error**: Check your credentials and internet connection
3. **Memory Issues**: Reduce batch size in training script for limited RAM
4. **Dependency Issues**: Ensure all packages in requirements.txt are installed

### Error Messages

- "Model not loaded": Check if the model training completed successfully
- "Reddit API credentials missing": Add your credentials to secrets.toml
- "Rate limit exceeded": Reddit API has rate limits; wait and try again

## Disclaimer

⚠️ **Important**: This tool is for educational and demonstration purposes only. It should not be used as a substitute for professional mental health advice, diagnosis, or treatment. If you or someone you know is experiencing mental health issues, please consult qualified mental health professionals.

## Contributing

This is an educational project. Feel free to modify and extend it for your learning purposes.

## License

This project is for educational use. Please respect the terms of service of all APIs and libraries used.

## Support

For issues related to:
- **Streamlit**: Check the [Streamlit documentation](https://docs.streamlit.io/)
- **Transformers**: Check the [Hugging Face documentation](https://huggingface.co/docs/transformers/)
- **Reddit API**: Check the [PRAW documentation](https://praw.readthedocs.io/)
- **LIME**: Check the [LIME documentation](https://lime-ml.readthedocs.io/)