from flask import Flask, request, jsonify, render_template
from rag import generate_answer

# --- Main Application Setup ---
app = Flask(__name__)

# --- Routes ---

@app.route("/")
def index():
    """Serves the main frontend UI."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    """API endpoint to answer questions based on the document knowledge base."""
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Delegate complex generation logic to the separated RAG module
    answer = generate_answer(question)
    
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)