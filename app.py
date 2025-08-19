import flask
import flask_cors
import os
import sentence_transformers
import torch

with open(file= os.path.join(os.getcwd(), "document", "chatbot_input_data.txt"), mode= "r", encoding= "utf-8") as file:
    text_data= file.read()

model = sentence_transformers.SentenceTransformer(model_name_or_path= "all-MiniLM-L6-v2")
# Context split into passages or chunks
chunks = text_data.split(". ")
passages = [chunk.strip() for chunk in chunks if chunk.strip() != ""]
# Embed context
passage_embeddings = model.encode(passages, convert_to_tensor=True)

app = flask.Flask(__name__)
flask_cors.CORS(app)

@app.route("/", methods=["GET", "POST"])
def home():
    return flask.render_template("index.html")
    
@app.route("/chat", methods=["POST"])
def chat():
    user_query = flask.request.json.get("message")
    if not user_query:
        return flask.jsonify({"response": "No input received."})
    
    # Encode question
    question_embedding = model.encode(user_query, convert_to_tensor= True)
    # Compute cosine similarity
    cos_scores = sentence_transformers.util.pytorch_cos_sim(question_embedding, passage_embeddings)[0]
    # Get best match
    top_result = torch.argmax(cos_scores)
    score = cos_scores[top_result]

    # Threshold-based filtering
    THRESHOLD = 0.5
    if score >= THRESHOLD:
        answer= passages[top_result]
    else:
        answer= "Sorry, I don't have an answer for this."
            
    return flask.jsonify({"response": answer})


if __name__ == "__main__":
    app.run(debug= True, use_reloader= False, threaded= True)



# ✅ Option A: Render.com (Free Plan)

# Create a free account

# Create a new "Web Service"

# Connect your GitHub repo or paste code manually

# Set:

# Runtime = Python

# Start command = python chatbot.py (or your file)

# It gives you a public URL like https://your-bot.onrender.com

# ✅ Option B: Replit.com

# Create a Replit project (Python + Flask)

# Paste the same chatbot code

# Click "Run"

# Enable the web server, get a URL like https://chatbot.username.repl.co