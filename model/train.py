"""
MindTrack Model Training Script
This script trains a DistilBERT model for mental health sentiment analysis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthDataset(Dataset):
    """Custom dataset class for mental health text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the mental health dataset.
    
    Args:
        filepath (str): Path to the CSV file containing the dataset
        
    Returns:
        tuple: Training and validation splits (X_train, X_val, y_train, y_val)
    """
    try:
        # Load the dataset
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Check if required columns exist
        if 'text' not in df.columns:
            raise ValueError("Dataset must contain 'text' column")
        
        # Check for different label column names
        if 'label' in df.columns:
            label_col = 'label'
        elif 'class' in df.columns:
            label_col = 'class'
        else:
            raise ValueError("Dataset must contain 'label' or 'class' column")
        
        # Map string labels to numeric if necessary
        if df[label_col].dtype == 'object':
            # Map suicide/non-suicide to 1/0, or risk/normal to 1/0
            label_mapping = {
                'suicide': 1, 'non-suicide': 0,
                'risk': 1, 'normal': 0,
                'Risk': 1, 'Normal': 0,
                'SUICIDE': 1, 'NON-SUICIDE': 0
            }
            
            if df[label_col].iloc[0] in label_mapping:
                df[label_col] = df[label_col].map(label_mapping)
                logger.info(f"Mapped labels: {dict(df[label_col].value_counts())}")
            else:
                logger.warning(f"Unknown label format: {df[label_col].unique()}")
        
        # Basic text cleaning
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].str.strip()
        df = df.dropna()
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0]
        
        # For large datasets, sample for training (increased sample size)
        if len(df) > 50000:
            logger.info(f"Large dataset detected ({len(df)} samples). Sampling 50,000 for training.")
            # Ensure balanced sampling
            label_counts = df[label_col].value_counts()
            if len(label_counts) == 2:  # Binary classification
                min_class_size = min(label_counts.values)
                if min_class_size >= 25000:  # If we have enough of each class
                    # Sample equally from each class
                    class_0 = df[df[label_col] == 0].sample(n=25000, random_state=42)
                    class_1 = df[df[label_col] == 1].sample(n=25000, random_state=42)
                    df = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
                    logger.info("Created balanced dataset with 25,000 samples per class")
                else:
                    # Use all available data
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    logger.info("Using all available data (insufficient for balanced 50k)")
            else:
                df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        elif len(df) > 20000:
            logger.info(f"Medium dataset detected ({len(df)} samples). Using all data.")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Label distribution:\n{df[label_col].value_counts()}")
        
        # Split the data
        X = df['text'].values
        y = df[label_col].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
        
    except FileNotFoundError:
        logger.error(f"File {filepath} not found. Creating sample data...")
        return create_sample_data()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes."""
    
    sample_texts = [
        "I'm feeling really happy today and excited about life!",
        "Everything seems pointless and I can't find motivation anymore",
        "Had a great day at work, feeling accomplished",
        "I feel so alone and nobody understands me",
        "Looking forward to the weekend with friends",
        "I can't stop crying and feel hopeless about everything",
        "Just got promoted at work, life is good!",
        "I hate myself and wish I could disappear",
        "Beautiful weather today, perfect for a walk",
        "Everything hurts and I don't want to get out of bed",
        "Celebrating my anniversary with my partner today",
        "I feel like a burden to everyone around me",
        "Excited about my vacation next week",
        "Nothing matters anymore, what's the point",
        "Had an amazing dinner with family",
        "I feel trapped and see no way out",
        "Learning new skills and growing every day",
        "Can't sleep, mind racing with negative thoughts",
        "Grateful for all the good things in my life",
        "Life feels empty and meaningless"
    ]
    
    # 0 = Normal, 1 = Risk
    sample_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        sample_texts, sample_labels, test_size=0.2, random_state=42
    )
    
    logger.info("Using sample data for demonstration")
    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

def compute_metrics(eval_pred):
    """
    Compute accuracy and F1-score for model evaluation.
    
    Args:
        eval_pred: Tuple containing predictions and labels
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Define model checkpoint
    model_checkpoint = 'distilbert-base-uncased'
    
    # Load and preprocess data
    data_path = 'data/cleaned_mental_health_data.csv'
    if not os.path.exists(data_path):
        # Fallback to original sample data
        data_path = 'data/mental_health_data.csv'
    
    X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train, tokenizer)
    val_dataset = MentalHealthDataset(X_val, y_val, tokenizer)
    
    # Load pre-trained model with proper initialization
    logger.info("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label={0: "NORMAL", 1: "RISK"},
        label2id={"NORMAL": 0, "RISK": 1},
        problem_type="single_label_classification",  # Explicit problem type
        ignore_mismatched_sizes=True  # Allow classifier head reinitialization
    )
    
    # Reinitialize the classifier head for better learning
    if hasattr(model, 'classifier'):
        torch.nn.init.normal_(model.classifier.weight, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias)
        logger.info("Reinitialized classifier head")
    
    model.to(device)
    
    # Move model to GPU if available
    model.to(device)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments optimized for better learning
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Increased epochs for better learning
        per_device_train_batch_size=8 if device.type == 'cuda' else 4,  # Smaller batch for better gradients
        per_device_eval_batch_size=8 if device.type == 'cuda' else 4,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,  # Explicit learning rate
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb logging
        fp16=device.type == 'cuda',  # Enable mixed precision for GPU
        dataloader_num_workers=2 if device.type == 'cuda' else 0,
        gradient_accumulation_steps=1,  # No gradient accumulation for now
        save_total_limit=3,
        seed=42,  # Set seed for reproducibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"Training on {len(train_dataset)} samples")
    logger.info(f"Validation on {len(val_dataset)} samples")
    
    trainer.train()
    
    # Save the model and tokenizer
    save_directory = "./saved_model"
    logger.info(f"Saving model to {save_directory}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    logger.info("Training completed successfully!")
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()