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
    
    if result['is_hallucination'] or result.get('confidence', 1.0) < 0.7:
        return f"⚠️ REVIEW REQUIRED\n\nAnswer: {answer}\n\nConfidence: {result.get('confidence', 'N/A'):.3f}\nRecommendation: {result['recommendation']}\nExpert Opinion: {result.get('expert_opinion', 'Pending review')}"
    else:
        return f"✅ SAFE ANSWER\n\nAnswer: {answer}\n\nConfidence: {result.get('confidence', 'N/A'):.3f}\nExpert Opinion: {result.get('expert_opinion', 'No review needed')}\nRecommendation: {result['recommendation']}"

# Usage
answer = medical_qa_with_detection("What is diabetes?")
print(answer)
