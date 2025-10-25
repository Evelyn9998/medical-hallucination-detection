# Reena: 
import torch.nn.functional as F
# from captum.attr import IntegratedGradients
# import shap
from lime.lime_text import LimeTextExplainer
import spacy
from IPython.display import display, HTML

nlp = spacy.load("en_core_sci_sm")
# end

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

    # Reena
    # def explain_with_tokens(self, question, answer, target_label=1, top_k=10):
    #     """
    #     Explain prediction with token-level attributions using Captum Integrated Gradients.
    #     """
    #     text = f"Question: {question} Answer: {answer}"

    #     # Tokenize input
    #     encoding = self.tokenizer(
    #         text,
    #         return_tensors='pt',
    #         truncation=True,
    #         padding='max_length',
    #         max_length=self.max_length
    #     )

    #     input_ids = encoding['input_ids'].to(self.device)
    #     attention_mask = encoding['attention_mask'].to(self.device)

    #     # Define proper forward function
    #     def forward_func(input_ids):
    #         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #         probs = F.softmax(outputs.logits, dim=-1)
    #         return probs[:, target_label]  # Return hallucination probability

    #     ig = IntegratedGradients(forward_func)

    #     # Compute attributions
    #     attributions, delta = ig.attribute(
    #         input_ids,
    #         n_steps=20,
    #         return_convergence_delta=True
    #     )

    #     # Sum across embedding dimensions
    #     attributions_sum = attributions.sum(dim=-1).squeeze(0)

    #     # Normalize and pair with tokens
    #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    #     scores = attributions_sum / torch.norm(attributions_sum)
    #     ranked = sorted(zip(tokens, scores.tolist()), key=lambda x: abs(x[1]), reverse=True)

    #     print("\n🩺 Top contributing tokens:")
    #     for token, score in ranked[:top_k]:
    #         print(f"{token:15s} -> {score:.4f}")

    #     return ranked


    # def explain_with_shap(self, question, answer):
    #     """
    #     Explain prediction using SHAP (model-agnostic).
    #     Generates a visual explanation for token contributions.
    #     """
    #     text = f"Question: {question} Answer: {answer}"

    #     # Define a prediction function that takes raw text input
    #     def f(texts):
    #         encodings = self.tokenizer(
    #             texts,
    #             return_tensors='pt',
    #             truncation=True,
    #             padding=True,
    #             max_length=self.max_length
    #         ).to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model(**encodings)
    #             probs = F.softmax(outputs.logits, dim=-1)
    #         return probs.cpu().numpy()

    #     # ✅ Pass only a list of strings, not a tokenizer
    #     explainer = shap.Explainer(f)
    #     shap_values = explainer([text])

    #     print("\n🧩 SHAP explanation generated. Opening interactive plot...")
    #     shap.plots.text(shap_values[0])


    # def explain_with_lime(self, question, answer):
    #     """
    #     Explain prediction using LIME (local interpretable model-agnostic explanation).
    #     """
    #     text = f"Question: {question} Answer: {answer}"

    #     # Create LIME text explainer
    #     explainer = LimeTextExplainer(class_names=["Not Hallucination", "Hallucination"])

    #     def predict_proba(texts):
    #         encodings = self.tokenizer(
    #             texts,
    #             return_tensors='pt',
    #             truncation=True,
    #             padding=True,
    #             max_length=self.max_length
    #         ).to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model(**encodings)
    #             probs = F.softmax(outputs.logits, dim=-1)
    #         return probs.cpu().numpy()

    #     # explanation = explainer.explain_instance(text, predict_proba, num_features=10)
    #     # print("\n🧩 LIME explanation:")
    #     # print(explanation.as_list())  # prints words + weights
    #     # explanation.show_in_notebook(text=True)
    #     explanation = explainer.explain_instance(text, predict_proba, num_features=10)

    #     print("\n🧩 LIME explanation (Top words):")
    #     for word, weight in explanation.as_list():
    #         print(f"{word:15s} -> {weight:.4f}")

    #     # For Jupyter or Colab, also display HTML
    #     try:
    #         from IPython.display import display
    #         display(explanation.show_in_notebook(text=True))
    #     except Exception:
    #         pass
    def explain_with_lime_medical(self, question, answer, num_features=10):
      """
      Explain prediction using LIME, highlighting medical entities.
      Non-evidence-based tokens contributing to hallucination are highlighted.
      """
      text = f"{question} {answer}"

      # Create LIME text explainer
      explainer = LimeTextExplainer(class_names=["Not Hallucination", "Hallucination"])

      # Prediction function for LIME
      def predict_proba(texts):
          encodings = self.tokenizer(
              texts,
              return_tensors='pt',
              truncation=True,
              padding=True,
              max_length=self.max_length
          ).to(self.device)
          with torch.no_grad():
              outputs = self.model(**encodings)
              probs = torch.softmax(outputs.logits, dim=-1)
          return probs.cpu().numpy()

      # Generate LIME explanation
      explanation = explainer.explain_instance(text, predict_proba, num_features=num_features)

      # Extract top contributing tokens
      top_tokens = [word for word, weight in explanation.as_list() if abs(weight) > 0]

      # Extract medical entities using scispaCy
      doc = nlp(text)
      medical_entities = [ent.text for ent in doc.ents]

      # Separate medical vs non-medical contributors
      medical_contrib = [t for t in top_tokens if t in medical_entities]
      non_medical_contrib = [t for t in top_tokens if t not in medical_entities]

      # print("\n🧩 LIME explanation (Top words):")
      # for word, weight in explanation.as_list():
      #     print(f"{word:15s} -> {weight:.4f}")

      if medical_contrib:
          print("\n🔍 Medical entities contributing to hallucination prediction:", medical_contrib)
      if non_medical_contrib:
          print("⚠️ Non-evidence-based terms contributing to hallucination prediction:", non_medical_contrib)
      
      if medical_contrib or non_medical_contrib:
          print("\n🩺 Contextual medical explanation:")

      if non_medical_contrib and medical_contrib:
          print(f"\n 💬 The model flagged this as a HALLUCINATION mainly because of non-evidence-based terms like "
                f"{', '.join([f'‘{w}’' for w in non_medical_contrib[:3]])}, "
                f"\n even though some medical terms such as "
                f"{', '.join([f'‘{w}’' for w in medical_contrib[:3]])} appeared.")
      
      elif non_medical_contrib:
          print(f"\n 💬 The model flagged this as a HALLUCINATION mainly due to non-evidence-based terms like "
                f"\n {', '.join([f'‘{w}’' for w in non_medical_contrib[:5]])}.")
      
      elif medical_contrib:
          print(f"\n 💬 The model’s decision was influenced by medical terms like "
                f"{', '.join([f'‘{w}’' for w in medical_contrib[:5]])}, "
                f"\n which may indicate domain relevance but uncertain factual grounding.")

      

      # # Optional: interactive colored visualization for notebooks
      # html_str = ""
      # for word, weight in explanation.as_list():
      #     if word in medical_contrib:
      #         color = "rgba(0,255,0,0.3)"  # evidence-based medical term
      #     elif word in non_medical_contrib:
      #         color = "rgba(255,0,0,0.3)"  # hallucination contributor
      #     else:
      #         color = "rgba(200,200,200,0.2)"
      #     html_str += f"<span style='background-color:{color}; padding:2px; margin:1px'>{word}</span> "

      # display(HTML(f"<div style='font-size:16px; line-height:1.6'>{html_str}</div>"))

    # end
    
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

            # Reena

            # explain = input("Would you like to explain this prediction? (y/n): ").strip().lower()
            # if explain == 'y':
            #     print("\nChoose method:")
            #     print("1. Token-level Attribution (Captum)")
            #     print("2. SHAP Explanation")
            #     print("3. LIME Explanation")
            #     choice = input("Enter choice (1/2/3): ").strip()

            #     if choice == '1':
            #         tester.explain_with_tokens(question, answer)
            #     elif choice == '2':
            #         tester.explain_with_shap(question, answer)
            #     elif choice == '3':
            #         tester.explain_with_lime(question, answer)
            #     else:
            #         print("Invalid choice.")
            auto_explain = True
            method_choice = "3"  # "1"=Captum, "2"=SHAP, "3"=LIME

            if auto_explain:
                print("\n🧠 Generating explanation...")
                if method_choice == "1":
                    tester.explain_with_tokens(question, answer)
                elif method_choice == "2":
                    tester.explain_with_shap(question, answer)
                elif method_choice == "3":
                    # tester.explain_with_lime(question, answer)
                    tester.explain_with_lime_medical(question, answer)
                # end
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()