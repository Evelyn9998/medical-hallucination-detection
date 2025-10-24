from llama_cpp import Llama
import requests

# Load local model
llm = Llama(
    model_path="D:\PrivateChatGPT\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Replace with actual path
    n_ctx=2048,
    n_threads=4
)


def medical_qa_with_detection(question):
    # Use local Mistral model
    prompt = f"[INST] {question} [/INST]"
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["[INST]", "</s>"]
    )
    answer = response['choices'][0]['text'].strip()
    
    # Check for hallucinations
    detection = requests.post('http://localhost:5001/detect',
        json={'question': question, 'answer': answer})
    result = detection.json()
    
    if result['is_hallucination'] or result.get('confidence', 1.0) < 0.7 or result.get('physiology_flag', False):
        return (
            f"⚠️ REVIEW REQUIRED\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Is Hallucination: {result['is_hallucination']}\n"
            f"Confidence: {result.get('confidence', 'N/A'):.3f}\n"
            f"Physiology Flag: {result.get('physiology_flag')}\n"
            f"Physiology Check: {result.get('note', 'No physiology note')}\n"
            f"Recommendation: {result.get('recommendation')}\n"
            f"Expert Opinion: {result.get('expert_opinion', 'Pending review')}\n"
        )
    else:
        return (
            f"✅ SAFE ANSWER\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Is Hallucination: {result['is_hallucination']}\n"
            f"Confidence: {result.get('confidence', 'N/A'):.3f}\n"
            f"Physiology Flag: {result.get('physiology_flag')}\n"
            f"Physiology Check: {result.get('note', 'Physiologically plausible')}\n"
            f"Expert Opinion: {result.get('expert_opinion', 'No review needed')}\n"
            f"Recommendation: {result['recommendation']}\n"
        )

# Usage
answer = medical_qa_with_detection("What happens to heart rate during exercise?")
print(answer)

