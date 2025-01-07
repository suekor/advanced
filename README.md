Ollama Chatbot with ChromaDB

This project demonstrates a chatbot powered by Ollama, which stores user queries and the corresponding responses in ChromaDB. It uses Streamlit for the frontend, allowing users to interact with the bot and view stored embeddings. You can run your own chatbot by following the steps below and work with AI in a user-friendly interface.

Installation

To run the chatbot application, follow the steps below:

Prerequisites
Python 3.x
Ollama API running locally on port 11434 (make sure Ollama is installed and running)
Streamlit for the frontend
ChromaDB to store the query/response history and embeddings
Steps to Install:
Create a virtual environment:
python -m venv venv
Activate the virtual environment:
On macOS/Linux:
source venv/bin/activate
On Windows:
venv\Scripts\activate
Download and install the required libraries:
pip install streamlit requests chromadb transformers torch
Make sure Ollama is running on your local machine.
You can download Ollama from the Ollama website and follow their setup instructions to run it on port 11434.
Run the application:
streamlit run chatbot.py
This will start the Streamlit app, which you can access in your browser.
Usage

Once the app is running, you can interact with the chatbot via the following features:

Enter your question:
Type any question into the input box and submit to get a response from Ollama.
View past interactions:
The chatbot keeps a history of all past queries and responses, stored in ChromaDB.
Search for past queries:
You can search through the stored queries and responses using the search input.
Example
User Input:
Hi

Ollama Response:
How can I assist you today?

Search Example:

0: "Hi"
1: "How can I assist you today?"
Project Structure

src/
│
├── chatbot.py         # Main chatbot application logic and Streamlit UI
└── chatbot_logic.py   # Logic for interaction with Ollama and ChromaDB

test/
│
└── test_chatbot.py    # Unit tests for the chatbot logic
Running Tests

To test the functionality of your chatbot, you can use unittest.

Run the tests with the following command:
python3 -m unittest discover test/
This will automatically discover and run all tests in the test/ directory.
Troubleshooting

Ollama not running:
Ensure that Ollama is properly installed and running locally on port 11434.
Error in Streamlit app:
If the app doesn't load properly, check the terminal for any error messages and ensure that all required libraries are installed.
