from flask import Flask, request, jsonify
import pickle
from sentencepiece import SentencePieceProcessor

#Load the model and docs
with open('TwoTowerModel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('docs_dict.pkl', 'rb') as f:
    docs_dict = pickle.load(f)

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

sp = sp.SentencePieceProcessor()
sp.load('model_1.model')

#Initialization
app = Flask(__name__)

#necessary function
def get_embedding(word, embeddings):
  if word in embeddings:
    return embeddings[word]
  else:
    return np.zeros(embeddings.vector_size)

def tokenize(text):
  text = text.lower().replace('\n','')
  tokens = sp.encode_as_pieces(text)
  return tokens

def preprocess_query(queries):
  query_embedding = []
  que_max_seq_length = 8
  for i in queries:
    query_embeddings = [get_embedding(token, embeddings) for token in tokenize(i)]
    query_embedding.append(query_embeddings)
  query_tensors = [torch.tensor(embedding).float() for embedding in query_embedding]
  query_padded = [F.pad(embedding, pad=(0, 0, 0, que_max_seq_length - embedding.size(0))) for embedding in query_tensors]
  query_batch = torch.stack(query_padded)
  return query_batch

def predict(query_encodings, docs_dict):
    k = 5
    # Calculate cosine similarity scores
    similarity_scores = [F.cosine_similarity(query_encodings.unsqueeze(0), docs_dict[doc_id].unsqueeze(0), dim=1) for doc_id in docs_dict.keys()]
    similarity_scores = torch.cat(similarity_scores, dim=1)
    # Rank documents by similarity
    ranked_documents = sorted(zip(docs_dict.keys(), similarity_scores[0]), key=lambda x: x[1], reverse=True)
    # Retrieve top-K documents
    top_k_documents = ranked_documents[:k]
    return top_k_documents


@app.route('/recommend', methods=['POST'])
def recommend():
    #Get user query
    try:
        user_query = request.json['query']
    except KeyError:
         return jsonify({'error': 'Missing "query" field in request body'}), 400

    #preprocess the query
    test_query_batch = preprocess_query(user_query)
    #prepare encodings
    query_encodings, _, _ = two_tower_model.forward(test_query_batch, None, None)
    #make prediction
    prediction = predict(query_encodings, docs_dict)
    return jsonify({'recommendations': prediction})

if __name__ == '__main__':
    app.run(debug=True)