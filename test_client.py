#!/usr/bin/env python3
"""
Simple Test Client for Medical Hallucination Detection API
Quick Testing
"""

import requests
import json
import time

class APITester:
    """Simple API testing client"""
    
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_health(self):
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Agent loaded: {data.get('agent_loaded')}")
                return True
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_single_detection(self):
        """Test single detection endpoint"""
        print("\n🔍 Testing single detection...")
        
        test_cases = [
            {
                "question": "What is diabetes?",
                "answer": "Diabetes is a chronic condition where blood sugar levels are persistently high due to insulin resistance or deficiency.",
                "expected": "Should be accurate"
            },
            {
                "question": "How to cure cancer?",
                "answer": "Cancer can be completely cured by drinking green tea and doing yoga every day.",
                "expected": "Should detect hallucination"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Question: {test_case['question']}")
            print(f"Answer: {test_case['answer']}")
            print(f"Expected: {test_case['expected']}")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/detect",
                    json={
                        "question": test_case["question"],
                        "answer": test_case["answer"]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = "HALLUCINATION" if result.get("is_hallucination") else "NORMAL"
                    confidence = result.get("confidence", 0)
                    recommendation = result.get("recommendation", "No recommendation")
                    
                    print(f"✅ Detection result: {status}")
                    print(f"📊 Confidence: {confidence:.3f}")
                    print(f"💡 Recommendation: {recommendation}")
                else:
                    print(f"❌ API call failed: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
    
    def test_batch_detection(self):
        """Test batch detection endpoint"""
        print("\n🔍 Testing batch detection...")
        
        qa_pairs = [
            {
                "question": "What causes heart attacks?",
                "answer": "Heart attacks are primarily caused by coronary artery blockage due to plaque buildup."
            },
            {
                "question": "How to treat pneumonia?",
                "answer": "Pneumonia is treated with antibiotics, rest, and supportive care."
            },
            {
                "question": "What is Alzheimer's disease?",
                "answer": "Alzheimer's is caused by 5G radiation and can be prevented by wearing aluminum foil hats."
            }
        ]
        
        try:
            response = self.session.post(
                f"{self.base_url}/batch_detect",
                json={"qa_pairs": qa_pairs}
            )
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                total = result.get("total_processed", 0)
                
                print(f"✅ Batch detection completed")
                print(f"📊 Total processed: {total}")
                
                for i, detection_result in enumerate(results, 1):
                    status = "HALLUCINATION" if detection_result.get("is_hallucination") else "NORMAL"
                    confidence = detection_result.get("confidence", 0)
                    
                    print(f"\n   Result {i}: {status} (confidence: {confidence:.3f})")
                    print(f"   Q: {detection_result.get('question', '')[:50]}...")
                    print(f"   A: {detection_result.get('answer', '')[:50]}...")
            else:
                print(f"❌ Batch detection failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Batch test failed: {e}")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎯 Interactive Testing Mode")
        print("=" * 40)
        print("Enter medical questions and answers to test the API")
        print("Type 'quit' to exit")
        
        while True:
            try:
                print("\n" + "-" * 40)
                question = input("Enter medical question: ").strip()
                if question.lower() == 'quit':
                    break
                
                answer = input("Enter answer to evaluate: ").strip()
                if answer.lower() == 'quit':
                    break
                
                print("\n🔍 Detecting...")
                response = self.session.post(
                    f"{self.base_url}/detect",
                    json={"question": question, "answer": answer}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = "HALLUCINATION ⚠️" if result.get("is_hallucination") else "NORMAL ✅"
                    confidence = result.get("confidence", 0)
                    recommendation = result.get("recommendation", "No recommendation")
                    
                    print(f"\n📊 Result: {status}")
                    print(f"🎯 Confidence: {confidence:.3f}")
                    print(f"💡 Recommendation: {recommendation}")
                else:
                    print(f"❌ API call failed: HTTP {response.status_code}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 MEDICAL HALLUCINATION DETECTION API TESTER")
        print("=" * 50)
        
        # Test health first
        if not self.test_health():
            print("\n❌ Health check failed. Make sure the API server is running!")
            print("💡 Start the server with: python api_server.py")
            return
        
        # Run detection tests
        self.test_single_detection()
        self.test_batch_detection()
        
        print("\n🎉 All tests completed!")
        print("\n💡 Want to try interactive testing? Run:")
        print("   python test_client.py --interactive")

def main():
    """Main function"""
    import sys
    
    tester = APITester()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        tester.interactive_test()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()
