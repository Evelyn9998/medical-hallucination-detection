#!/usr/bin/env python3
"""
Test script for the trained Medical Hallucination Detection model
Run this after training the model with the main script
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class MedicalHallucinationTester:
    """Class for testing the trained hallucination detection model"""
    
    def __init__(self, model_path="./med_hallucination_model", max_length=512):
        """
        Initialize the tester with a trained model
        
        Args:
            model_path: Path to the saved model directory
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model and tokenizer
        try:
            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained and saved the model first!")
            raise
    
    def predict_single(self, question, answer, return_probability=True):
        """
        Predict whether a single answer is a hallucination
        
        Args:
            question: Medical question
            answer: Answer to evaluate
            return_probability: Whether to return probability score
            
        Returns:
            Dictionary with prediction results
        """
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
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probs = torch.softmax(outputs.logits, dim=-1)
            hallucination_prob = probs[0][1].item()
        
        result = {
            'question': question,
            'answer': answer,
            'is_hallucination': bool(prediction),
            'confidence': hallucination_prob if prediction else (1 - hallucination_prob),
            'recommendation': self._get_recommendation(prediction, hallucination_prob)
        }
        
        if return_probability:
            result['hallucination_probability'] = hallucination_prob
            result['not_hallucination_probability'] = 1 - hallucination_prob
        
        return result
    
    def _get_recommendation(self, prediction, hallucination_prob):
        """Generate recommendation based on prediction and confidence"""
        if prediction == 1:  # Hallucination detected
            if hallucination_prob > 0.9:
                return "🚨 HIGH RISK: Strong hallucination detected - Do not use this answer"
            elif hallucination_prob > 0.7:
                return "⚠️  MEDIUM RISK: Potential hallucination - Requires expert review"
            else:
                return "❓ LOW CONFIDENCE: Uncertain prediction - Manual verification needed"
        else:  # No hallucination
            confidence = 1 - hallucination_prob
            if confidence > 0.9:
                return "✅ SAFE: Answer appears medically accurate"
            elif confidence > 0.7:
                return "👍 LIKELY SAFE: Answer seems reasonable but consider verification"
            else:
                return "❓ UNCERTAIN: Low confidence - Additional review recommended"
    
    def batch_predict(self, qa_pairs):
        """
        Predict hallucinations for multiple question-answer pairs
        
        Args:
            qa_pairs: List of tuples [(question1, answer1), (question2, answer2), ...]
            
        Returns:
            List of prediction results
        """
        results = []
        print(f"Processing {len(qa_pairs)} question-answer pairs...")
        
        for i, (question, answer) in enumerate(qa_pairs):
            result = self.predict_single(question, answer)
            results.append(result)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(qa_pairs)} pairs")
        
        return results
    
    def run_demo_tests(self):
        """Run a series of demo tests with various medical questions"""
        
        test_cases = [
            # Correct medical answers
            ("What is the main cause of type 2 diabetes?", 
             "Insulin resistance and relative insulin deficiency, often associated with obesity and genetic factors"),
            
            ("What is the first-line treatment for hypertension?", 
             "Lifestyle modifications including diet, exercise, and weight management, followed by ACE inhibitors or thiazide diuretics"),
            
            ("What are the symptoms of pneumonia?", 
             "Cough, fever, chest pain, shortness of breath, and sometimes chills and fatigue"),
            
            # Hallucinated medical answers
            ("What causes heart attacks?", 
             "Heart attacks are primarily caused by eating too much sugar and drinking cold water"),
            
            ("How is cancer treated?", 
             "Cancer can be completely cured by drinking green tea and doing yoga every day"),
            
            ("What is Alzheimer's disease?", 
             "Alzheimer's is caused by 5G radiation and can be prevented by wearing aluminum foil hats"),
            
            # Ambiguous or partially incorrect answers
            ("What causes diabetes?", 
             "Diabetes is caused by lifestyle factors and sometimes genetics"),
            
            ("How to treat a fever?", 
             "Take aspirin and drink lots of fluids, fever always indicates serious infection"),
        ]
        
        print("\n" + "="*80)
        print("MEDICAL HALLUCINATION DETECTION - DEMO TESTS")
        print("="*80)
        
        for i, (question, answer) in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            result = self.predict_single(question, answer)
            
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Prediction: {'HALLUCINATION' if result['is_hallucination'] else 'NOT HALLUCINATION'}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Recommendation: {result['recommendation']}")
            
            if 'hallucination_probability' in result:
                print(f"Hallucination Probability: {result['hallucination_probability']:.3f}")
        
        print("\n" + "="*80)

def main():
    """Main function to run the tests"""
    
    # Check if model exists
    import os
    model_path = "./med_hallucination_model"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run the training script first to create the model!")
        return
    
    # Initialize tester
    try:
        tester = MedicalHallucinationTester(model_path)
    except Exception as e:
        print(f"Failed to initialize tester: {e}")
        return
    
    # Run demo tests
    tester.run_demo_tests()
    
    # Interactive testing
    print("\n" + "="*80)
    print("INTERACTIVE TESTING")
    print("="*80)
    print("Enter your own medical questions and answers to test!")
    print("Type 'quit' to exit")
    
    while True:
        try:
            question = input("\nEnter medical question: ").strip()
            if question.lower() == 'quit':
                break
                
            answer = input("Enter answer to evaluate: ").strip()
            if answer.lower() == 'quit':
                break
            
            result = tester.predict_single(question, answer)
            
            print(f"\nResult: {'HALLUCINATION' if result['is_hallucination'] else 'NOT HALLUCINATION'}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Recommendation: {result['recommendation']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()