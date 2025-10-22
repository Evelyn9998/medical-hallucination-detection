#!/usr/bin/env python3
"""
Flask API Server for Medical Hallucination Detection
Simple and Fast Integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import HallucinationDetectionAgent

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the hallucination detection agent"""
    global agent
    try:
        agent = HallucinationDetectionAgent()
        logger.info("Hallucination Detection Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'agent_loaded': agent is not None,
        'message': 'Medical Hallucination Detection API is running'
    })

@app.route('/detect', methods=['POST'])
def detect_hallucination():
    """Single hallucination detection endpoint"""
    try:
        if agent is None:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        
        if not question or not answer:
            return jsonify({'error': 'Both question and answer are required'}), 400
        
        # Perform detection
        result = agent.detect(question, answer)
        
        # Add API response metadata
        result['api_version'] = '1.0'
        result['timestamp'] = datetime.now().isoformat()
        result['status'] = 'success'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_hallucination: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect_hallucination():
    """Batch hallucination detection endpoint"""
    try:
        if agent is None:
            return jsonify({'error': 'Agent not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        qa_pairs = data.get('qa_pairs', [])
        if not qa_pairs:
            return jsonify({'error': 'qa_pairs list is required'}), 400
        
        # Validate data format
        formatted_pairs = []
        for i, pair in enumerate(qa_pairs):
            if not isinstance(pair, dict) or 'question' not in pair or 'answer' not in pair:
                return jsonify({
                    'error': f'Invalid format for pair {i}. Expected dict with question and answer keys'
                }), 400
            formatted_pairs.append((pair['question'], pair['answer']))
        
        # Perform batch detection
        results = agent.batch_detect(formatted_pairs)
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'api_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in batch_detect_hallucination: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    docs = {
        'title': 'Medical Hallucination Detection API',
        'version': '1.0',
        'description': 'REST API for detecting hallucinations in medical question-answer pairs',
        'base_url': 'http://localhost:5001',
        'endpoints': {
            'GET /health': {
                'description': 'Health check endpoint',
                'response': 'JSON with status, timestamp, and agent status'
            },
            'POST /detect': {
                'description': 'Detect hallucination in a single Q&A pair',
                'request_body': {
                    'question': 'string (required) - Medical question',
                    'answer': 'string (required) - Answer to evaluate'
                },
                'response': {
                    'question': 'Original question',
                    'answer': 'Original answer',
                    'is_hallucination': 'boolean - True if hallucination detected',
                    'confidence': 'float - Confidence score (0-1)',
                    'recommendation': 'string - Safety recommendation',
                    'api_version': 'string - API version',
                    'timestamp': 'string - ISO timestamp',
                    'status': 'string - success/error'
                }
            },
            'POST /batch_detect': {
                'description': 'Detect hallucinations in multiple Q&A pairs',
                'request_body': {
                    'qa_pairs': 'array of objects - Each object must have question and answer fields'
                },
                'response': {
                    'results': 'array - Detection results for each pair',
                    'total_processed': 'int - Number of pairs processed',
                    'api_version': 'string - API version',
                    'timestamp': 'string - ISO timestamp',
                    'status': 'string - success/error'
                }
            },
            'GET /api/docs': {
                'description': 'This documentation endpoint',
                'response': 'JSON with API documentation'
            }
        },
        'examples': {
            'single_detection': {
                'request': {
                    'method': 'POST',
                    'url': '/detect',
                    'headers': {'Content-Type': 'application/json'},
                    'body': {
                        'question': 'What causes diabetes?',
                        'answer': 'Diabetes is caused by insulin resistance and genetic factors.'
                    }
                },
                'response': {
                    'question': 'What causes diabetes?',
                    'answer': 'Diabetes is caused by insulin resistance and genetic factors.',
                    'is_hallucination': False,
                    'confidence': 0.85,
                    'recommendation': '✅ SAFE: Answer appears medically accurate',
                    'api_version': '1.0',
                    'timestamp': '2025-10-22T10:30:00.123456',
                    'status': 'success'
                }
            },
            'batch_detection': {
                'request': {
                    'method': 'POST',
                    'url': '/batch_detect',
                    'headers': {'Content-Type': 'application/json'},
                    'body': {
                        'qa_pairs': [
                            {
                                'question': 'What causes diabetes?',
                                'answer': 'Diabetes is caused by insulin resistance.'
                            },
                            {
                                'question': 'How to treat fever?',
                                'answer': 'Take aspirin and rest.'
                            }
                        ]
                    }
                },
                'response': {
                    'results': [
                        {
                            'question': 'What causes diabetes?',
                            'answer': 'Diabetes is caused by insulin resistance.',
                            'is_hallucination': False,
                            'confidence': 0.82,
                            'recommendation': '✅ SAFE: Answer appears medically accurate'
                        },
                        {
                            'question': 'How to treat fever?',
                            'answer': 'Take aspirin and rest.',
                            'is_hallucination': False,
                            'confidence': 0.75,
                            'recommendation': '👍 LIKELY SAFE: Answer seems reasonable but consider verification'
                        }
                    ],
                    'total_processed': 2,
                    'api_version': '1.0',
                    'timestamp': '2025-10-22T10:30:00.123456',
                    'status': 'success'
                }
            }
        },
        'curl_examples': {
            'health_check': 'curl -X GET http://localhost:5001/health',
            'single_detection': '''curl -X POST http://localhost:5001/detect \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is hypertension?",
    "answer": "Hypertension is high blood pressure, a chronic condition where blood pressure is consistently elevated."
  }' ''',
            'batch_detection': '''curl -X POST http://localhost:5001/batch_detect \\
  -H "Content-Type: application/json" \\
  -d '{
    "qa_pairs": [
      {
        "question": "What causes heart attacks?",
        "answer": "Heart attacks are primarily caused by coronary artery blockage."
      },
      {
        "question": "How is cancer treated?",
        "answer": "Cancer treatment includes surgery, chemotherapy, and radiation therapy."
      }
    ]
  }' '''
        }
    }
    return jsonify(docs)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with basic info"""
    return jsonify({
        'name': 'Medical Hallucination Detection API',
        'version': '1.0',
        'status': 'running',
        'agent_loaded': agent is not None,
        'endpoints': ['/health', '/detect', '/batch_detect', '/api/docs'],
        'documentation': '/api/docs'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 MEDICAL HALLUCINATION DETECTION API")
    print("=" * 60)
    print("🚀 Starting Flask API server...")
    
    # Initialize agent
    if initialize_agent():
        print("✅ Agent initialized successfully")
        print("🌐 Server will start at: http://localhost:5001")
        print("📚 API Documentation: http://localhost:5001/api/docs")
        print("❤️  Health Check: http://localhost:5001/health")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ Failed to start server due to agent initialization failure")
        print("💡 Make sure your model is trained and saved in './med_hallucination_model'")
