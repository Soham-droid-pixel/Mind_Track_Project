"""
MindTrack Model Training Script - Robust Version
This script trains a DistilBERT model for mental health sentiment analysis
with best practices to prevent overfitting and ensure stable training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import logging

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MentalHealthDataset(Dataset):
    """Custom dataset class for mental health text classification with improved handling."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of corresponding labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (reduced from 512 to prevent overfitting)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with truncation and padding
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
    Compute comprehensive metrics for model evaluation.
    
    Args:
        eval_pred: Tuple containing predictions and labels
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Calculate per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Calculate macro F1 for balanced evaluation
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'f1_macro': f1_macro,
        'precision_normal': precision[0] if len(precision) > 0 else 0.0,
        'recall_normal': recall[0] if len(recall) > 0 else 0.0,
        'f1_normal': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'precision_risk': precision[1] if len(precision) > 1 else 0.0,
        'recall_risk': recall[1] if len(recall) > 1 else 0.0,
        'f1_risk': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
    }

def main():
    """Main training function with robust hyperparameters to prevent overfitting."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Define model checkpoint
    model_checkpoint = 'distilbert-base-uncased'
    
    # Load and preprocess data
    data_path = 'data/cleaned_mental_health_data.csv'
    if not os.path.exists(data_path):
        # Fallback to original sample data
        data_path = 'data/mental_health_data.csv'
    
    X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path)
    
    # Log dataset statistics
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Training class distribution: {np.bincount(y_train)}")
    logger.info(f"Validation class distribution: {np.bincount(y_val)}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets with proper max length
    max_length = 256  # Reduced from 512 to prevent overfitting to very long sequences
    train_dataset = MentalHealthDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_dataset = MentalHealthDataset(X_val, y_val, tokenizer, max_length=max_length)
    
    # Load pre-trained model with proper configuration
    logger.info("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label={0: "NORMAL", 1: "RISK"},
        label2id={"NORMAL": 0, "RISK": 1},
        problem_type="single_label_classification",
        # Note: DistilBERT dropout is configured in the config, not as init parameters
    )
    
    # Initialize classifier head with smaller standard deviation for stability
    if hasattr(model, 'classifier'):
        torch.nn.init.normal_(model.classifier.weight, std=0.01)  # Smaller std
        torch.nn.init.zeros_(model.classifier.bias)
        logger.info("Reinitialized classifier head with smaller variance")
    
    model.to(device)
    
    # Data collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    # Calculate optimal batch size and steps
    batch_size = 16 if device.type == 'cuda' else 8
    total_train_samples = len(train_dataset)
    steps_per_epoch = total_train_samples // batch_size
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    
    # CRITICAL: Robust TrainingArguments to prevent overfitting
    training_args = TrainingArguments(
        # Output and logging
        output_dir='./results',
        logging_dir='./logs',
        logging_steps=50,
        
        # Training schedule - KEY FOR PREVENTING OVERFITTING
        num_train_epochs=4,  # Moderate number of epochs
        learning_rate=2e-5,  # Conservative learning rate for stable training
        lr_scheduler_type='cosine',  # Smooth decay prevents abrupt changes
        warmup_ratio=0.1,  # Warm up over first 10% of training
        
        # Batch sizes
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        
        # Regularization - CRITICAL FOR GENERALIZATION
        weight_decay=0.01,  # L2 regularization
        max_grad_norm=1.0,  # Gradient clipping prevents exploding gradients
        
        # Evaluation and saving - PREVENTS OVERFITTING
        evaluation_strategy="epoch",  # Evaluate at end of each epoch
        save_strategy="epoch",  # Save at end of each epoch
        load_best_model_at_end=True,  # CRITICAL: Load best model, not last
        metric_for_best_model="f1_macro",  # Use macro F1 for balanced evaluation
        greater_is_better=True,  # Higher F1 is better
        
        # Early stopping patience
        save_total_limit=3,  # Keep only best 3 checkpoints
        
        # Performance optimizations
        fp16=device.type == 'cuda',  # Mixed precision for GPU
        dataloader_num_workers=2 if device.type == 'cuda' else 0,
        dataloader_pin_memory=device.type == 'cuda',
        
        # Reproducibility
        seed=42,
        data_seed=42,
        
        # Disable external logging
        report_to=None,
        
        # Additional stability settings
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        prediction_loss_only=False,
        
        # Memory management
        dataloader_drop_last=False,
        remove_unused_columns=True,
    )
    
    # Early stopping callback for additional overfitting protection
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # Stop if no improvement for 2 epochs
        early_stopping_threshold=0.001  # Minimum improvement threshold
    )
    
    # Create trainer with comprehensive configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],  # Add early stopping
    )
    
    # Log training configuration
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_checkpoint}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Max epochs: {training_args.num_train_epochs}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Weight decay: {training_args.weight_decay}")
    logger.info(f"Max sequence length: {max_length}")
    logger.info("=" * 60)
    
    # Train the model
    logger.info("🚀 Starting training with overfitting prevention...")
    
    try:
        trainer.train()
        logger.info("✅ Training completed successfully!")
        
        # Get final evaluation results
        logger.info("📊 Evaluating best model...")
        final_eval = trainer.evaluate()
        
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("=" * 60)
        for key, value in final_eval.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return False
    
    # Save the best model and tokenizer
    save_directory = "./saved_model"
    logger.info(f"💾 Saving best model to {save_directory}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    try:
        # Save model and tokenizer
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        
        # Save training metadata
        metadata = {
            'model_checkpoint': model_checkpoint,
            'max_length': max_length,
            'final_eval': final_eval,
            'training_samples': len(train_dataset),
            'validation_samples': len(val_dataset)
        }
        
        import json
        with open(os.path.join(save_directory, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("✅ Model and metadata saved successfully!")
        
        # Final verification - test the saved model
        logger.info("🧪 Verifying saved model...")
        test_model_loading(save_directory)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving model: {str(e)}")
        return False

def test_model_loading(model_path):
    """Test that the saved model loads correctly and produces reasonable outputs."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        # Load saved model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        # Test with sample texts
        test_texts = [
            "I'm feeling really happy and optimistic about life!",
            "I can't see any point in living anymore, everything is hopeless"
        ]
        
        logger.info("Testing model with sample texts:")
        for text in test_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Prediction: {prediction} ({'RISK' if prediction == 1 else 'NORMAL'})")
            logger.info(f"Confidence: {confidence:.4f}")
            logger.info(f"Logits: {logits.tolist()[0]}")
            logger.info("-" * 40)
        
        logger.info("✅ Model verification completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {str(e)}")
        
if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("You can now use the model with: python test_model.py")
    else:
        logger.error("❌ Training pipeline failed!")
        exit(1)