import unittest
from unittest.mock import patch, MagicMock
from src.ap2 import get_ollama_response, add_to_chromadb, search_in_chromadb, collection


class TestChatbot(unittest.TestCase):
    @patch("src.ap2.requests.post")
    def test_get_ollama_response_success(self, mock_post):
        # Мок успешного ответа от Ollama API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response

        result = get_ollama_response("Test prompt")
        self.assertEqual(result, "Test response")

    @patch("src.ap2.requests.post")
    def test_get_ollama_response_error(self, mock_post):
        # Мок ошибки от Ollama API
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = get_ollama_response("Test prompt")
        self.assertIn("Error Ollama API: 500", result)

def test_add_to_chromadb(self):
    # Очистим коллекцию перед тестом
    collection.delete(where={})  # Удаляем все документы в коллекции

    user_query = "What is AI?"
    ollama_response = "AI stands for Artificial Intelligence."

    add_to_chromadb(user_query, ollama_response)

    # Проверяем, что данные добавлены
    data = collection.get()
    self.assertEqual(len(data["documents"]), 2)
    self.assertIn(user_query, data["documents"])
    self.assertIn(ollama_response, data["documents"])

def test_search_in_chromadb(self):
    # Очистим коллекцию и добавим тестовые данные
    collection.delete(where={})  # Удаляем все документы в коллекции
    user_query = "What is AI?"
    ollama_response = "AI stands for Artificial Intelligence."
    add_to_chromadb(user_query, ollama_response)

    # Ищем по ключевому слову
    results = search_in_chromadb("AI")
    self.assertGreater(len(results), 0)
    self.assertIn(user_query, results)
    self.assertIn(ollama_response, results)



if __name__ == "__main__":
    unittest.main()
