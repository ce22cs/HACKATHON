from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    Loads the pre-trained DialoGPT model and tokenizer.
    """
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

def get_bot_response(user_input):
    """
    Generates a chatbot response using the DialoGPT model.
    """
    try:
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
        
        with torch.no_grad():  # Disable gradient calculation
            response = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        
        bot_reply = tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return bot_reply
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint to handle user messages and return chatbot responses.
    """
    data = request.get_json()
    user_message = data.get("message") if data else None

    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    bot_reply = get_bot_response(user_message)
    return jsonify({"reply": bot_reply})

@app.route("/")
def home():
    """
    Health check endpoint to verify if the server is running.
    """
    return jsonify({"status": "Chatbot is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
