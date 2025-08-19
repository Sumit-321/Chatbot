import flask
import flask_cors
import os
import sentence_transformers
import torch

with open(file= os.path.join(os.getcwd(), "chatbot_input_data.txt"), mode= "r", encoding= "utf-8") as file:
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
    return "", 200
    
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


