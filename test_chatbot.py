import unittest
from src.ap2 import get_ollama_response, add_to_chromadb, search_in_chromadb

class TestChatbot(unittest.TestCase):

    def test_ollama_response(self):
        # Проверка ответа от Ollama
        prompt = "What is AI?"
        response = get_ollama_response(prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_add_to_chromadb(self):
        # Проверка добавления данных в ChromaDB
        user_query = "What is Python?"
        ollama_response = "Python is a programming language."
        try:
            add_to_chromadb(user_query, ollama_response)
            result = True
        except Exception:
            result = False
        self.assertTrue(result)

    def test_search_in_chromadb(self):
        # Проверка поиска в ChromaDB
        query = "Python"
        results = search_in_chromadb(query)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
