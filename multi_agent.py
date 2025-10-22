import openai
import requests


# === Agent 1: AnswerAgent ===
def answer_agent(question):
    """Generate a medical answer using the LLM."""
    print("🧠 [AnswerAgent] Generating medical answer...")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": question}]
    )
    answer = response.choices[0].message.content
    return answer


# === Agent 2: DetectionAgent ===
def detection_agent(question, answer):
    """Call the hallucination detection model API."""
    print("🔍 [DetectionAgent] Checking for hallucinations...")
    response = requests.post(
        "http://localhost:5001/detect",
        json={"question": question, "answer": answer}
    )
    result = response.json()
    # Expected structure: {"is_hallucination": bool, "confidence": float, "recommendation": str}
    return result




# === Agent 3: HumanReviewAgent ===
def human_review_agent(question, answer, detection_result):
    """Perform human or simulated expert review for uncertain or risky results."""
    print("👩‍⚕️ [HumanReviewAgent] Reviewing answer (non-high-confidence case)...")
    review_prompt = f"""
    The model produced the following medical answer:
    {answer}

    The hallucination detection system reported:
    {detection_result}

    Please act as a medical expert and determine whether this answer is accurate.
    If it contains hallucinations or inaccuracies, provide a corrected version or suggestions.
    """
    review = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": review_prompt}]
    )
    return review.choices[0].message.content


# === Coordinator ===
def coordinator(question):
    """Main control logic for coordinating between agents."""
    print(f"\n🚀 [Coordinator] Received question: {question}\n")

    # Step 1: Generate answer
    answer = answer_agent(question)
    print("🧾 Model Answer:\n", answer)

    # Step 2: Run hallucination detection
    detection_result = detection_agent(question, answer)
    print("🔬 Detection Result:\n", detection_result)

    confidence = detection_result.get("confidence", 0)
    is_hallucination = detection_result.get("is_hallucination", False)

    # === Step 3: Decision logic ===
    # ✅ SAFE (not hallucination & confidence > 0.9): Directly accept
    if not is_hallucination and confidence > 0.9:
        print("\n✅ SAFE: Answer appears medically accurate.")
        return f"✅ SAFE — {answer}"

    # 🚨 HIGH RISK (confidence > 0.9): Strong hallucination - reject and review
    elif is_hallucination and confidence > 0.9:
        print("\n🚨 HIGH RISK: Strong hallucination detected. Rejecting answer.")
        reviewed = human_review_agent(question, answer, detection_result)
        return f"🚨 HIGH RISK — Reviewed and corrected:\n{reviewed}"

    # ⚠️ MEDIUM RISK (0.7 < confidence ≤ 0.9): Potential hallucination - expert review required
    elif is_hallucination and 0.7 < confidence <= 0.9:
        print("\n⚠️ MEDIUM RISK: Potential hallucination. Sending to expert review.")
        reviewed = human_review_agent(question, answer, detection_result)
        return f"⚠️ MEDIUM RISK — Expert-reviewed answer:\n{reviewed}"

    # ❓ LOW CONFIDENCE (confidence ≤ 0.7): Uncertain prediction - manual verification needed
    elif confidence <= 0.7:
        print("\n❓ LOW CONFIDENCE: Uncertain detection. Sending to manual review.")
        reviewed = human_review_agent(question, answer, detection_result)
        return f"❓ LOW CONFIDENCE — Manually verified answer:\n{reviewed}"

    # 👍 LIKELY SAFE (moderate non-hallucination confidence): still reviewed for safety
    else:
        print("\n👍 LIKELY SAFE: Reasonable answer, but still under review for confirmation.")
        reviewed = human_review_agent(question, answer, detection_result)
        return f"👍 LIKELY SAFE — Human-reviewed answer:\n{reviewed}"


# === Example usage ===
if __name__ == "__main__":
    result = coordinator("What is diabetes?")
    print("\n🎯 Final Output:\n", result)