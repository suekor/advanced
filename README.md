# advanced
Ollama Chatbot with ChromaDB

This project demonstrates a chatbot powered by Ollama, which stores user queries and the corresponding responses in ChromaDB. It uses Streamlit for the frontend, allowing users to interact with the bot and view stored embeddings. You can run your own chatbot by turning it on in terminal and work with ai in a better interface

Installation
To run the chatbot application, follow the steps below:

Prerequisites
- Python 3.x
- Ollama API running locally on port `11434`
- Streamlit for the frontend
- ChromaDB to store the query/response history and embeddings

Steps to install:
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2.Activate a virtual envoirement:
   ```bash
 source venv/bin/activate
   ```
3.Download all required libraries:
pip install streamlit requests chromadb transformers torch

4. Make sure Ollama is running on your local machine. You can download it from the Ollama website and follow their setup instructions.
5. Run the application:
   ```bash
   streamlit run chatbot.py
   ```
   This will start the Streamlit app, which you can access in your browser.
Usage
- **Enter your question:** Type any question into the input box and submit to get a response from Ollama.
- **View past interactions:** The chatbot keeps a history of all past queries and responses, stored in ChromaDB.
- **Search for past queries:** You can search through the stored queries and responses using the search input.

Examples
1. **User Input:**
Hi

   **Ollama Response:**
   How can I assist you today?

2. **Search Example:**
[

0

:

"Hi"

1

:

"How can I assist you today?"

] '


src/
src/
│
├── app.py                
└── chatbot_logic.py      

test/
test/
│
├── test_chatbot.py       
├── test_chromadb.py      
└── test_app.py           
