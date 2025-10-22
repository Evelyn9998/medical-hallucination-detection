import sys
sys.path.append('.')

from test_model import MedicalHallucinationTester
from physiology_check import physiological_plausibility_check

class HallucinationDetectionAgent:
    """
    A simple agent for detecting hallucinations in medical answers.
    """

    def __init__(self, model_path="./med_hallucination_model"):
        """
        Initialize the agent with the trained model.

        Args:
            model_path (str): Path to the saved hallucination detection model directory.
        """
        self.tester = MedicalHallucinationTester(model_path)

    # def detect(self, question, answer):
    #     """
    #     Detect if the given answer is a hallucination for the question.

    #     Args:
    #         question (str): The medical question.
    #         answer (str): The answer to evaluate.

    #     Returns:
    #         dict: Detection result including prediction, confidence, and recommendation.
    #     """
    #     return self.tester.predict_single(question, answer)


    def detect(self, question, answer):
      result = self.tester.predict_single(question, answer)  # your existing model prediction
      physiology_result = physiological_plausibility_check(question, answer)
      result.update(physiology_result)
      return result


    def batch_detect(self, qa_pairs):
        """
        Detect hallucinations for multiple question-answer pairs.

        Args:
            qa_pairs (list): List of tuples [(question1, answer1), ...]

        Returns:
            list: List of detection results.
        """
        return self.tester.batch_predict(qa_pairs)

def main():
    """
    Demo usage of the agent.
    """
    print("Initializing Hallucination Detection Agent...")
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
    print(f"Recommendation: {result['recommendation']}")

    # Another example
    print("\n--- Another Example ---")
    question2 = "What causes diabetes?"
    answer2 = "Eating too much sugar and not exercising."

    result2 = agent.detect(question2, answer2)
    print(f"Question: {result2['question']}")
    print(f"Answer: {result2['answer']}")
    print(f"Is Hallucination: {result2['is_hallucination']}")
    print(f"Confidence: {result2['confidence']:.3f}")
    print(f"Recommendation: {result2['recommendation']}")

if __name__ == "__main__":
    main()