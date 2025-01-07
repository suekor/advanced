import streamlit as st
import requests
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader  # Add dependency for PDF handling
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

    # Add user query and response as separate entries in ChromaDB
    existing_ids = collection.get().get("ids", [])
    query_id = f"query-{len(existing_ids) + 1}"
    response_id = f"response-{len(existing_ids) + 2}"

    collection.add(
        ids=[query_id, response_id],
        embeddings=[query_embeddings, response_embeddings],
        documents=[user_query, ollama_response],
        metadatas=[{"type": "query"}, {"type": "response"}]
    )
    st.success("Query and response added to ChromaDB.")

def process_uploaded_files(uploaded_files):
    """
    Process uploaded files, split their content using LangChain, and store in ChromaDB.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                add_document_to_chromadb(f"{uploaded_file.name}-chunk-{i+1}", chunk)
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() for page in reader.pages])
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                add_document_to_chromadb(f"{uploaded_file.name}-chunk-{i+1}", chunk)

def add_document_to_chromadb(doc_name, content):
    """
    Add document content to ChromaDB with embeddings.
    """
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        doc_embeddings = model(**inputs).last_hidden_state.mean(dim=1)[0].tolist()

    # Check existing IDs in collection
    existing_ids = collection.get().get("ids", [])
    document_id = f"{doc_name}-{len(existing_ids) + 1}"  # Unique ID for document

    # Add document to ChromaDB
    collection.add(
        ids=[document_id],
        embeddings=[doc_embeddings],
        documents=[content],
        metadatas=[{"name": doc_name}]
    )
    st.success(f"Document chunk '{doc_name}' added to ChromaDB.")

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

        if results and results["documents"]:
            return results["documents"]
        else:
            return ["No matches found."]
    except Exception as e:
        return [f"Error searching in ChromaDB: {e}"]

def answer_question_about_documents(question):
    results = search_in_chromadb(question)
    if results:
        st.write("Top results:")
        for i, doc in enumerate(results, 1):
            st.write(f"{i}. {doc}")
    else:
        st.write("No relevant documents found.")

def show_chromadb_history():
    try:
        # Use search_in_chromadb to fetch all stored queries and responses
        results = collection.get()
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents and metadatas:
            st.write("History of queries and responses:")
            for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                doc_type = meta.get("type", "unknown")
                st.write(f"{i}. [{doc_type}] {doc}")
        else:
            st.write("")
    except Exception as e:
        st.write(f"Error retrieving history: {e}")

st.title("Ollama Chatbot with LangChain and ChromaDB")

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

# File uploader
uploaded_files = st.file_uploader("Upload documents (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
if uploaded_files:
    process_uploaded_files(uploaded_files)

# Question about uploaded documents
question = st.text_input("Ask a question about uploaded documents:")
if question:
    answer_question_about_documents(question)
