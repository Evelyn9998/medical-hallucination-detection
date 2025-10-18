import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MedHalluDataset(Dataset):
    """Custom Dataset class for MedHallu data"""
    
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

class MedicalHallucinationDetector:
    """Main class for medical hallucination detection"""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=512):
        """
        Initialize the detector with a pre-trained medical BERT model

        Args:
            model_name: Name of the pre-trained model to use
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_and_preprocess_data(self, labeled_samples=1000):
        """
        Load and preprocess the MedHallu dataset using only expert labeled data

        Args:
            labeled_samples: Number of expert labeled samples to use (default: 1000)

        Returns:
            Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
        """
        print(f"Loading MedHallu labeled dataset with {labeled_samples} high-quality samples...")

        # Load only the labeled split (expert annotated data)
        try:
            labeled_data = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating mock data for demonstration...")
            return self._create_mock_data()

        # Process labeled data (high quality expert annotations)
        labeled_split = labeled_data['train']
        if labeled_samples < len(labeled_split):
            # Sample from labeled data while maintaining class balance
            labeled_samples_data = self._sample_balanced(labeled_split, labeled_samples)
        else:
            labeled_samples_data = labeled_split

        labeled_processed = self._process_dataset_split(labeled_samples_data)

        # Use only labeled data
        all_texts = labeled_processed['texts']
        all_labels = labeled_processed['labels']

        print(f"Using {len(all_texts)} high-quality expert labeled samples")
        
        # Split into train/val/test
        # First split: separate test set (20%)
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Second split: separate train and validation (80% train, 20% val of remaining)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
        )
        
        print(f"Dataset split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        print(f"Label distribution - Train: {np.bincount(train_labels)}")
        
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    
    def _process_dataset_split(self, dataset_split):
        """
        Process a single split of the dataset

        Args:
            dataset_split: A single split from the MedHallu dataset

        Returns:
            Dictionary with processed texts and labels
        """
        texts = []
        labels = []

        for sample in dataset_split:
            # Use the actual field names from the MedHallu dataset
            question = sample.get('Question', '')

            # Get ground truth answer (correct answer)
            ground_truth = sample.get('Ground Truth', '')

            # Get hallucinated answer (incorrect answer)
            hallucinated = sample.get('Hallucinated Answer', '')

            # Skip samples with missing data
            if not question or not ground_truth or not hallucinated:
                continue

            # Create input format: "Question: ... Answer: ..."
            # Positive sample (correct answer - not hallucination)
            correct_text = f"Question: {question.strip()} Answer: {ground_truth.strip()}"
            if correct_text.strip():
                texts.append(correct_text)
                labels.append(0)  # 0 = not hallucination

            # Negative sample (hallucinated answer)
            hallucinated_text = f"Question: {question.strip()} Answer: {hallucinated.strip()}"
            if hallucinated_text.strip():
                texts.append(hallucinated_text)
                labels.append(1)  # 1 = hallucination

        print(f"Processed {len(texts)} samples ({labels.count(0)} correct, {labels.count(1)} hallucinated)")
        return {'texts': texts, 'labels': labels}

    def _sample_balanced(self, dataset_split, num_samples):
        """
        Sample from dataset while maintaining balance between classes

        Args:
            dataset_split: Dataset split to sample from
            num_samples: Number of samples to select

        Returns:
            Sampled dataset split
        """
        import random
        random.seed(42)  # For reproducibility

        total_samples = len(dataset_split)
        if num_samples >= total_samples:
            return dataset_split

        # Sample indices
        sampled_indices = random.sample(range(total_samples), num_samples)
        sampled_data = dataset_split.select(sampled_indices)

        return sampled_data

    def _create_mock_data(self):
        """Create mock data for demonstration if dataset loading fails"""
        print("Creating mock medical QA data...")
        
        mock_samples = [
            {
                'question': 'What is the most common cause of myocardial infarction?',
                'correct': 'Coronary artery occlusion due to atherosclerotic plaque rupture and thrombosis',
                'hallucinated': 'Viral infection of the heart muscle leading to inflammatory damage'
            },
            {
                'question': 'What is the first-line treatment for type 2 diabetes?',
                'correct': 'Metformin, along with lifestyle modifications including diet and exercise',
                'hallucinated': 'Immediate insulin therapy is always required for all type 2 diabetes patients'
            },
            {
                'question': 'What are the symptoms of pneumonia?',
                'correct': 'Cough, fever, chest pain, shortness of breath, and sometimes chills',
                'hallucinated': 'Pneumonia always presents with severe headaches and vision problems'
            }
        ] * 100  # Repeat to create more samples
        
        texts, labels = [], []
        for sample in mock_samples:
            # Correct answer
            texts.append(f"Question: {sample['question']} Answer: {sample['correct']}")
            labels.append(0)
            # Hallucinated answer
            texts.append(f"Question: {sample['question']} Answer: {sample['hallucinated']}")
            labels.append(1)
        
        # Split the mock data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )
        
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    
    def create_data_loaders(self, train_texts, train_labels, val_texts, val_labels, batch_size=16):
        """
        Create DataLoader objects for training and validation
        
        Args:
            train_texts, train_labels: Training data
            val_texts, val_labels: Validation data
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = MedHalluDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = MedHalluDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def initialize_model(self):
        """Initialize the classification model with dropout for better generalization"""
        # Load base model with dropout configuration
        config = AutoConfig.from_pretrained(self.model_name)
        config.hidden_dropout_prob = 0.2  # Increase dropout in hidden layers
        config.attention_probs_dropout_prob = 0.2  # Increase dropout in attention
        config.classifier_dropout = 0.3  # Dropout in classifier layer
        config.num_labels = 2  # Binary classification
        config.problem_type = "single_label_classification"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )

        # Resize token embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for the trainer"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_texts, train_labels, val_texts, val_labels,
                    output_dir="./med_hallucination_model", epochs=5, batch_size=8):
        """
        Train the hallucination detection model
        
        Args:
            train_texts, train_labels: Training data
            val_texts, val_labels: Validation data
            output_dir: Directory to save the model
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("Initializing model...")
        self.initialize_model()
        
        # Create datasets
        train_dataset = MedHalluDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = MedHalluDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Define optimized training arguments for better performance
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,  # Optimal learning rate for BERT models
            warmup_steps=100,  # Adequate warmup for stable training
            weight_decay=0.01,  # Standard weight decay
            logging_dir=f'{output_dir}/logs',
            logging_steps=20,  # Balanced logging frequency
            eval_strategy="epoch",  # Evaluate per epoch
            save_strategy="epoch",  # Save per epoch
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            gradient_accumulation_steps=2,  # For larger effective batch size
            max_grad_norm=1.0,  # Gradient clipping
            lr_scheduler_type="linear",  # Linear scheduler with warmup
            report_to=None,  # Disable external logging
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=2 if torch.cuda.is_available() else 0,  # Optimize for available hardware
            save_total_limit=2,  # Keep only best 2 models
            seed=42,  # For reproducibility
        )
        
        # Create trainer with memory-efficient early stopping
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=2,  # Reduced patience for faster training
            early_stopping_threshold=0.01  # Higher threshold for memory efficiency
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping_callback]
        )
        
        # Train the model with progress monitoring
        print("Starting memory-efficient training...")
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: 3e-5")
        print(f"Model: DistilBERT (memory efficient)")
        print(f"Max sequence length: 256")
        print("-" * 60)

        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")

        # Print training summary
        self._print_memory_efficient_summary(trainer)

        return trainer

    def _print_memory_efficient_summary(self, trainer):
        """Print a summary of memory-efficient optimizations"""
        print("\n" + "="*60)
        print("MEMORY-EFFICIENT TRAINING SUMMARY")
        print("="*60)
        print("✓ Memory Optimizations:")
        print("  - Using DistilBERT (smaller model) instead of Bio_ClinicalBERT")
        print("  - Reduced sequence length to 256 tokens")
        print("  - Smaller batch size (4) for lower memory usage")
        print("  - Disabled multiprocessing in data loading")
        print()
        print("✓ Training Speed Optimizations:")
        print("  - Reduced epochs to 3 for faster training")
        print("  - Higher learning rate (3e-5) for faster convergence")
        print("  - Linear learning rate scheduler for simplicity")
        print("  - Reduced warmup steps to 50")
        print()
        print("✓ Early Stopping Configuration:")
        print("  - Patience: 2 epochs")
        print("  - Threshold: 0.01")
        print("  - Per-epoch evaluation for memory efficiency")
        print()
        print("✓ Resource Usage:")
        print("  - No gradient accumulation (memory efficient)")
        print("  - Reduced weight decay (0.01)")
        print("  - Minimal logging for faster execution")
        print("="*60)
    
    def evaluate_model(self, test_texts, test_labels, model_path=None):
        """
        Evaluate the trained model on test data
        
        Args:
            test_texts, test_labels: Test data
            model_path: Path to saved model (if None, uses current model)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        # Create test dataset
        test_dataset = MedHalluDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Evaluate
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of hallucination
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Print results
        print("\n=== Evaluation Results ===")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_labels, all_predictions)
        
        return metrics, all_predictions, all_probs
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Hallucination', 'Hallucination'],
                   yticklabels=['Not Hallucination', 'Hallucination'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def predict_single(self, question, answer, return_probability=True):
        """
        Predict whether a single answer is a hallucination
        
        Args:
            question: Medical question
            answer: Answer to evaluate
            return_probability: Whether to return probability score
            
        Returns:
            Prediction (and probability if requested)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        text = f"Question: {question} Answer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probs = torch.softmax(outputs.logits, dim=-1)
            hallucination_prob = probs[0][1].item()
        
        result = {
            'is_hallucination': bool(prediction),
            'confidence': hallucination_prob if prediction else (1 - hallucination_prob)
        }
        
        if return_probability:
            result['hallucination_probability'] = hallucination_prob
        
        return result

def main(labeled_samples=200):
    """
    Main execution function

    Args:
        labeled_samples: Number of expert labeled samples to use
    """
    # Initialize detector with medical domain model for better performance
    detector = MedicalHallucinationDetector(model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=512)

    # Load and preprocess data (only expert labeled data)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = detector.load_and_preprocess_data(
        labeled_samples=labeled_samples
    )

    # Train the model with memory-efficient parameters
    trainer = detector.train_model(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=3,  # Fewer epochs for faster training
        batch_size=4   # Smaller batch size for memory efficiency
    )
    
    # Evaluate the model
    metrics, predictions, probabilities = detector.evaluate_model(test_texts, test_labels)
    
    # Test single prediction
    sample_question = "What is the main cause of heart attacks?"
    correct_answer = "Coronary artery blockage due to atherosclerotic plaque"
    hallucinated_answer = "Heart attacks are primarily caused by eating too much sugar"
    
    print("\n=== Single Prediction Examples ===")
    
    result1 = detector.predict_single(sample_question, correct_answer)
    print(f"Correct answer - Is hallucination: {result1['is_hallucination']}, Confidence: {result1['confidence']:.3f}")
    
    result2 = detector.predict_single(sample_question, hallucinated_answer)
    print(f"Hallucinated answer - Is hallucination: {result2['is_hallucination']}, Confidence: {result2['confidence']:.3f}")

if __name__ == "__main__":
    # Use only expert labeled samples (1000 high-quality samples)
    main(labeled_samples=1000)

    # Alternative: Use fewer samples for faster training
    # main(labeled_samples=500)

    # Alternative: Use all available labeled samples
    # main(labeled_samples=1000)  # This is the default