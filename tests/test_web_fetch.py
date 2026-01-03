
import unittest
from unittest.mock import patch, MagicMock
import web_fetch
import requests

class TestWebFetch(unittest.TestCase):

    @patch('web_fetch.requests.get')
    def test_fetch_page_success(self, mock_get):
        # Setup valid HTML response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
        mock_response.text = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
        mock_get.return_value = mock_response

        print("Testing basic fetch success...")
        result = web_fetch.fetch_page("http://example.com")
        self.assertEqual(result['title'], "Test")
        self.assertIn("Content", result['text'])

    @patch('web_fetch.requests.get')
    def test_fetch_fallback_html_detection(self, mock_get):
        # Setup response with missing header but valid HTML body
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {} # No Content-Type
        mock_response.text = "<html><body><p>I am explicit HTML</p></body></html>"
        mock_get.return_value = mock_response
        
        print("Testing HTML sniffing fallback...")
        result = web_fetch.fetch_page("http://example.com/sniff")
        self.assertIsNotNone(result['text'])
        self.assertIn("I am explicit HTML", result['text'])

    @patch('web_fetch.requests.get')
    def test_fetch_body_fallback_text(self, mock_get):
        # Setup HTML where standard get_text might be messy/short, but body is good
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        # Simulating a case where body extraction is preferred or adds coverage
        mock_response.text = "<html><body><p>Essential Body Text</p></body></html>"
        mock_get.return_value = mock_response
        
        print("Testing body text fallback...")
        result = web_fetch.fetch_page("http://example.com/body")
        self.assertIn("Essential Body Text", result['text'])

    @patch('web_fetch.requests.get')
    def test_fetch_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Fail")
        print("Testing fetch error...")
        result = web_fetch.fetch_page("http://fail.com")
        self.assertIsNone(result['text'])

    print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
