# Advanced Ollama Chatbot with ChromaDB

This project demonstrates a chatbot powered by Ollama, which stores user queries and corresponding responses in ChromaDB. It also supports uploading documents (text and PDF) for enhanced interaction, with added functionality to process and query data related to specific articles from the Constitution. The application uses Streamlit for the frontend, allowing users to interact with the bot, view stored embeddings, and manage historical data.

### Installation

To run the chatbot application, follow these steps:

#### Prerequisites

- Python 3.x
- Ollama API running locally on port `11434`
- Streamlit for the frontend
- ChromaDB to store the query/response history and embeddings
- Additional dependencies for PDF and text processing

#### Steps to Install:

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   # For Linux/macOS:
   source venv/bin/activate

   # For Windows:
   venv\Scripts\activate
   ```

3. **Install all required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running locally:**
   Download Ollama from its website and follow their setup instructions to run the API on your machine.

5. **Run the application:**
   ```bash
   streamlit run ap3.py
   ```
   This will start the Streamlit app, which you can access in your browser.

### Usage

1. **Ask Questions:**
   - Type a question in the input box and receive a response from Ollama.

2. **Upload Documents:**
   - Upload `.txt` or `.pdf` files. Their content will be split into smaller chunks and stored in ChromaDB with embeddings.

3. **Search History:**
   - Search through the stored queries and responses or uploaded document content using the search bar.

4. **Ask About Uploaded Documents:**
   - Use the chatbot to retrieve relevant information from the uploaded documents.

5. **Query Constitution Articles:**
   - Users can now load a Constitution document and query specific articles for relevant information. Ensure to ask questions before loading the Constitution to optimize interactions.

### Examples

#### Basic Interaction

**User Input:**
```text
Hi
```

**Ollama Response:**
```text
How can I assist you today?
```

#### Search Example
Stored results in ChromaDB:
```
0: "Hi"
1: "How can I assist you today?"
```

#### Document Processing
Upload a PDF or text file, and it will be chunked, embedded, and stored for future interactions.

#### Constitution Query Example

1. Load the Constitution using the `Load Constitution` button.
2. Ask a question like:
   ```text
   Article 2
   ```
3. Receive a response based on the loaded Constitution.

### File Structure

```plaintext
src/
├── ap3.py                # Streamlit app and main UI logic


test/
├── test_chatbot.py       # Unit tests for chatbot logic

