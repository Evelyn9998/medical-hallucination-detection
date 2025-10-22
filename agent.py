import sys
sys.path.append('.')

from test_model import MedicalHallucinationTester
from expert_reviewer import ExpertReviewer

class HallucinationDetectionAgent:
    """
    Agent for detecting hallucinations in medical answers,
    with optional expert review for low-confidence or flagged cases.
    """

    def __init__(self, model_path="./med_hallucination_model"):
        """
        Initialize the agent with the trained model and expert reviewer.

        Args:
            model_path (str): Path to the saved hallucination detection model directory.
        """
        self.tester = MedicalHallucinationTester(model_path)
        self.reviewer = ExpertReviewer()

    def detect(self, question, answer):
        """
        Detect if the given answer is a hallucination for the question.
        If confidence < 0.7 or hallucination detected, trigger expert review.

        Args:
            question (str): The medical question.
            answer (str): The answer to evaluate.

        Returns:
            dict: Detection result including prediction, confidence, recommendation, and expert review info.
        """
        detection = self.tester.predict_single(question, answer)

        # Step 3: 判断是否需要专家审核
        if detection["is_hallucination"] or detection["confidence"] < 0.7:
            expert_opinion = self.reviewer.review(question, answer, detection)
            result = {
                **detection,
                "review_required": True,
                "expert_opinion": expert_opinion,
                "recommendation": "⚠️ Review required by medical expert"
            }
        else:
            result = {
                **detection,
                "review_required": False,
                "expert_opinion": "✅ Confidence sufficient, no review needed",
                "recommendation": "Answer considered safe"
            }

        return result

    def batch_detect(self, qa_pairs):
        """
        Detect hallucinations for multiple question-answer pairs.

        Args:
            qa_pairs (list): List of tuples [(question1, answer1), ...]

        Returns:
            list: List of detection results.
        """
        results = []
        for question, answer in qa_pairs:
            results.append(self.detect(question, answer))
        return results

def main():
    """
    Demo usage of the agent with expert review logic.
    """
    print("Initializing Hallucination Detection Agent with Expert Reviewer...")
    agent = HallucinationDetectionAgent()

    # Example detection
    print("\n--- Example Detection ---")
    question = "What is the main cause of heart attacks?"
    answer = "Coronary artery blockage due to plaque buildup."

    result = agent.detect(question, answer)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Review Required: {result['review_required']}")
    print(f"Expert Opinion: {result['expert_opinion']}")

    # Another example
    print("\n--- Another Example ---")
    question2 = "What causes diabetes?"
    answer2 = "Eating too much sugar and not exercising."

    result2 = agent.detect(question2, answer2)
    print(f"Question: {result2['question']}")
    print(f"Answer: {result2['answer']}")
    print(f"Is Hallucination: {result2['is_hallucination']}")
    print(f"Confidence: {result2['confidence']:.3f}")
    print(f"Review Required: {result2['review_required']}")
    print(f"Expert Opinion: {result2['expert_opinion']}")

if __name__ == "__main__":
    main()