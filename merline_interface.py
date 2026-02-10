"""
Merline Interface - Simple web interface for voice communication
Similar to modern voice assistant interfaces
"""

import os
import sys
import threading
import time
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Client

app = Flask(__name__)
CORS(app)

# Global Merline client
merline_client = None
conversation_history = []

# HTML Template for the interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merline - Voice Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #333;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 90%;
            max-width: 600px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .status {
            padding: 10px 20px;
            background: #f5f5f5;
            text-align: center;
            font-size: 12px;
            color: #666;
        }
        
        .status.active {
            background: #4CAF50;
            color: white;
        }
        
        .status.listening {
            background: #2196F3;
            color: white;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fafafa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.merline .message-bubble {
            background: #e0e0e0;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        
        .message-label {
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .controls {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .mic-button {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            margin: 0 auto;
            display: block;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
        }
        
        .mic-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .mic-button.recording {
            background: #f44336;
            animation: pulse 1s infinite;
        }
        
        .info {
            text-align: center;
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Merline</h1>
            <p>Modular Ethical Responsive Local Intelligent Neural Entity</p>
        </div>
        <div class="status" id="status">Initializing...</div>
        <div class="chat-area" id="chatArea">
            <div class="message merline">
                <div>
                    <div class="message-label">Merline</div>
                    <div class="message-bubble">Welcome! I'm ready to assist you. Click the microphone to start talking.</div>
                </div>
            </div>
        </div>
        <div class="controls">
            <button class="mic-button" id="micButton" onclick="toggleListening()">🎤</button>
            <div class="info">Click to start/stop listening</div>
        </div>
    </div>
    
    <script>
        let isListening = false;
        let statusCheckInterval;
        
        function addMessage(text, isUser) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'merline');
            
            const label = isUser ? 'You' : 'Merline';
            messageDiv.innerHTML = `
                <div>
                    <div class="message-label">${label}</div>
                    <div class="message-bubble">${text}</div>
                </div>
            `;
            
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function updateStatus(status, className = '') {
            const statusEl = document.getElementById('status');
            statusEl.textContent = status;
            statusEl.className = 'status ' + className;
        }
        
        function toggleListening() {
            fetch('/api/toggle', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    isListening = data.listening;
                    const button = document.getElementById('micButton');
                    if (isListening) {
                        button.classList.add('recording');
                        updateStatus('Listening...', 'listening');
                    } else {
                        button.classList.remove('recording');
                        updateStatus('Ready', 'active');
                    }
                });
        }
        
        function checkStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (data.listening && !isListening) {
                        document.getElementById('micButton').classList.add('recording');
                        updateStatus('Listening...', 'listening');
                        isListening = true;
                    } else if (!data.listening && isListening) {
                        document.getElementById('micButton').classList.remove('recording');
                        updateStatus('Ready', 'active');
                        isListening = false;
                    }
                });
        }
        
        function loadHistory() {
            fetch('/api/history')
                .then(r => r.json())
                .then(data => {
                    const chatArea = document.getElementById('chatArea');
                    chatArea.innerHTML = '';
                    data.history.forEach(msg => {
                        addMessage(msg.content, msg.role === 'user');
                    });
                });
        }
        
        // Check status every second
        statusCheckInterval = setInterval(checkStatus, 1000);
        
        // Load history on start
        loadHistory();
        setInterval(loadHistory, 2000);
        
        // Initial status check
        checkStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    if merline_client:
        return jsonify({
            'listening': merline_client.listening,
            'speaking': merline_client.is_speaking
        })
    return jsonify({'listening': False, 'speaking': False})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    if merline_client:
        merline_client.toggleListening()
        return jsonify({'listening': merline_client.listening})
    return jsonify({'listening': False})

@app.route('/api/history')
def history():
    if merline_client:
        history_list = [
            {'role': msg.role, 'content': msg.content}
            for msg in merline_client.history[-20:]  # Last 20 messages
        ]
        return jsonify({'history': history_list})
    return jsonify({'history': []})

def run_merline():
    """Run Merline in background"""
    global merline_client
    try:
        merline_client = Client(startListening=False, history=[])
        print("Merline initialized for web interface")
    except Exception as e:
        print(f"Error initializing Merline: {e}")

if __name__ == '__main__':
    # Start Merline in background thread
    merline_thread = threading.Thread(target=run_merline, daemon=True)
    merline_thread.start()
    
    # Wait a bit for Merline to initialize
    time.sleep(3)
    
    print("\n" + "="*60)
    print("Merline Web Interface")
    print("="*60)
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
