import streamlit as st
import requests
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize ChromaDB client
chromadb.api.client.SharedSystemClient.clear_system_cache()
client = chromadb.Client()

# Initialize models for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize or create ChromaDB collection
collection = client.get_or_create_collection(name="chatbot_data")

def get_ollama_response(prompt):
    # Set Ollama API endpoint and authentication headers
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama3.2:latest",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error Ollama API: {response.status_code}. Response:{response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama API: {e}"

def add_to_chromadb(user_query, ollama_response):
    # Generate embeddings for the user query
    inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_embeddings = model(**inputs).last_hidden_state.mean(dim=1)[0].tolist()

    # Generate embeddings for the Ollama response
    inputs_response = tokenizer(ollama_response, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        response_embeddings = model(**inputs_response).last_hidden_state.mean(dim=1)[0].tolist()

    # Check existing IDs in collection
    existing_ids = collection.get().get("ids", [])
    document_id_query = f"doc-query-{len(existing_ids) + 1}"  # Unique ID for user query
    document_id_response = f"doc-response-{len(existing_ids) + 1}"  # Unique ID for Ollama response

    # Add both the user query and the Ollama response to ChromaDB
    collection.add(
        ids=[document_id_query, document_id_response],
        embeddings=[query_embeddings, response_embeddings],
        documents=[user_query, ollama_response]
    )
    print(f"Text added to ChromaDB with IDs: {document_id_query}, {document_id_response}")


def search_in_chromadb(query):
    try:
        # Generate embeddings for the query
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            query_embeddings = model(**inputs).last_hidden_state.mean(dim=1)[0].tolist()
        
        # Query ChromaDB using query embeddings
        results = collection.query(
            query_embeddings=query_embeddings, 
            n_results=5  # Number of search results to return
        )
        
        if results["documents"]:
            return results["documents"]
        else:
            return ["No matches found."]
    except Exception as e:
        return [f"Error searching in ChromaDB: {e}"]

def show_chromadb_embeddings():
    try:
        # Retrieve all embeddings stored in ChromaDB
        embeddings = collection.get().get("embeddings", [])
        if embeddings:
            st.write("Stored embeddings:")
            for i, embedding in enumerate(embeddings, 1):
                st.write(f"{i}. {embedding[:10]}...")  # Display the first 10 dimensions for brevity
        else:
            st.write("No embeddings found.")
    except Exception as e:
        st.write(f"Error retrieving embeddings: {e}")



def show_chromadb_history():
    # Create a placeholder for the output
    placeholder = st.empty()
    try:
        # Retrieve all documents stored in ChromaDB
        all_docs = collection.get().get("documents", [])
        if all_docs:
            # Replace placeholder with the history
            with placeholder.container():
                st.write("History of queries and responses:")
                for i, doc in enumerate(all_docs, 1):
                    st.write(f"{i}. {doc}")
        else:
            placeholder.write("")
    except Exception as e:
        placeholder.write(f"Error retrieving history: {e}")



st.title("Ollama Chatbot with ChromaDB")


# Display saved history
show_chromadb_history()

# User input for new question
user_input = st.text_input("Enter your question:")
if user_input:
    response = get_ollama_response(user_input)
    st.write(f"Ollama Response: {response}")
    add_to_chromadb(user_input, response)

# Search functionality for past queries
search_query = st.text_input("Search history:", "")
if search_query:
    search_results = search_in_chromadb(search_query)
    st.write("Search results:")
    for result in search_results:
        st.write(result)