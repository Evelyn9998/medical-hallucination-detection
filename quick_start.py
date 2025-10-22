#!/usr/bin/env python3
"""
Quick Start Script for Medical Hallucination Detection API
English Version - Get Running in 3 Steps
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_header():
    """Print startup header"""
    print("=" * 60)
    print("🏥 MEDICAL HALLUCINATION DETECTION API")
    print("📚 Quick Start Guide")
    print("=" * 60)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if model exists
    model_path = Path("./med_hallucination_model")
    if not model_path.exists():
        print("❌ ERROR: Trained model not found!")
        print(f"   Expected location: {model_path.absolute()}")
        print("💡 Please train your model first before starting the API")
        return False
    
    # Check if required files exist
    required_files = ["agent.py", "test_model.py"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ ERROR: Required file not found: {file}")
            return False
    
    print("✅ All requirements met!")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    required_packages = ["flask", "flask-cors", "requests", "torch", "transformers"]
    
    try:
        for package in required_packages:
            print(f"   Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try installing manually: pip install flask flask-cors requests torch transformers")
        return False

def start_api_server():
    """Start the API server"""
    print("\n🚀 Starting API server...")
    
    # Check if api_server_en.py exists
    if not Path("api_server_en.py").exists():
        print("❌ ERROR: api_server_en.py not found!")
        print("💡 Make sure you have the API server file in the current directory")
        return None
    
    try:
        # Start the server process
        process = subprocess.Popen([
            sys.executable, "api_server_en.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for i in range(20):  # Wait up to 20 seconds
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server started successfully!")
                    return process
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            print(f"   Waiting... ({i+1}/20)")
        
        print("❌ Server startup timeout")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_api():
    """Test the API with a simple request"""
    print("\n🧪 Testing API...")
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:5000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Health check failed")
            return False
        
        # Test detection endpoint
        test_data = {
            "question": "What is diabetes?",
            "answer": "Diabetes is a chronic condition characterized by high blood sugar levels."
        }
        
        response = requests.post(
            "http://localhost:5000/detect",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            status = "HALLUCINATION" if result.get("is_hallucination") else "NORMAL"
            confidence = result.get("confidence", 0)
            
            print("✅ API test successful!")
            print(f"   Test result: {status} (confidence: {confidence:.3f})")
            return True
        else:
            print(f"❌ API test failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("🎯 YOUR API IS READY!")
    print("=" * 60)
    
    print("\n🌐 API Endpoints:")
    print("   • Health Check: http://localhost:5000/health")
    print("   • Documentation: http://localhost:5000/api/docs")
    print("   • Single Detection: POST http://localhost:5000/detect")
    print("   • Batch Detection: POST http://localhost:5000/batch_detect")
    
    print("\n📝 Quick Test Commands:")
    print("\n1. Test with curl:")
    print('''curl -X POST http://localhost:5000/detect \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What causes high blood pressure?",
    "answer": "High blood pressure is caused by various factors including genetics, diet, and lifestyle."
  }' ''')
    
    print("\n2. Test with Python:")
    print('''import requests

response = requests.post('http://localhost:5000/detect', 
    json={
        'question': 'What is hypertension?',
        'answer': 'Hypertension is persistently high blood pressure.'
    })

print(response.json())''')
    
    print("\n3. Test with our test client:")
    print("   python test_client_en.py")
    print("   python test_client_en.py --interactive")

def main():
    """Main function - Quick start process"""
    print_header()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed, but continuing...")
        print("   You may need to install packages manually if the server fails to start")
    
    # Step 3: Start server
    server_process = start_api_server()
    if not server_process:
        print("\n❌ Failed to start API server")
        return
    
    # Step 4: Test API
    if not test_api():
        print("\n⚠️  API test failed, but server is running")
        print("   You can still try to use the API manually")
    
    # Step 5: Show usage examples
    show_usage_examples()
    
    print(f"\n🎉 Setup complete! Your API is running at http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep the script running
        print("\n⏳ Server is running... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        print("✅ Server stopped. Goodbye!")

if __name__ == "__main__":
    main()
