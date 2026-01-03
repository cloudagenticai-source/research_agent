
import unittest
from unittest.mock import patch, MagicMock
import os
import web_search

class TestWebSearch(unittest.TestCase):
    
    @patch('web_search.GoogleSearch')
    @patch.dict(os.environ, {"SERPAPI_API_KEY": "fake_key"})
    def test_search_web_success(self, mock_google_search):
        # Setup mock response
        mock_instance = mock_google_search.return_value
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "Test Title 1",
                    "link": "http://example.com/1",
                    "snippet": "Snippet 1",
                    "source": "Example 1",
                    "position": 1
                },
                {
                    "title": "Test Title 2",
                    "link": "http://example.com/2",
                    "snippet": "Snippet 2",
                    "source": "Example 2",
                    "position": 2
                }
            ]
        }

        print("Testing search_web success...")
        results = web_search.search_web("test query", num_results=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], "Test Title 1")
        self.assertEqual(results[1]['link'], "http://example.com/2")
        mock_google_search.assert_called_with({
            "q": "test query",
            "num": 2,
            "api_key": "fake_key",
            "engine": "google"
        })

    @patch('web_search.GoogleSearch')
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self, mock_google_search):
        print("Testing missing API key...")
        with self.assertRaises(RuntimeError) as cm:
            web_search.search_web("test")
        self.assertIn("SERPAPI_API_KEY environment variable is not set", str(cm.exception))

    @patch('web_search.GoogleSearch')
    @patch.dict(os.environ, {"SERPAPI_API_KEY": "fake_key"})
    def test_api_error(self, mock_google_search):
        print("Testing API error...")
        mock_instance = mock_google_search.return_value
        mock_instance.get_dict.return_value = {"error": "Invalid API key"}
        
        with self.assertRaises(RuntimeError) as cm:
            web_search.search_web("test")
        self.assertIn("SerpAPI returned error: Invalid API key", str(cm.exception))

    print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
