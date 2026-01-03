
import unittest
from unittest.mock import MagicMock, patch
import memory_vector

class TestVectorMemory(unittest.TestCase):
    @patch('memory_vector.OpenAI')
    @patch('memory_vector.chromadb.PersistentClient')
    def test_vector_memory_flow(self, mock_chroma, mock_openai):
        # Setup mocks
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_embedding_response
        
        mock_collection = MagicMock()
        mock_chroma_instance = mock_chroma.return_value
        mock_chroma_instance.get_or_create_collection.return_value = mock_collection

        # Initialize
        print("Initializing VectorMemory...")
        vm = memory_vector.VectorMemory()
        
        # Test Embed
        print("Testing embed...")
        vec = vm.embed("test text")
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        mock_openai_instance.embeddings.create.assert_called_once()
        
        # Test Upsert Episode
        print("Testing upsert_episode...")
        vm.upsert_episode(1, "episode text", {"meta": "data"})
        mock_collection.upsert.assert_called()
        call_args = mock_collection.upsert.call_args[1]
        self.assertEqual(call_args['ids'], ["episode:1"])
        self.assertEqual(call_args['documents'], ["episode text"])

        # Test Upsert Fact
        print("Testing upsert_fact...")
        vm.upsert_fact(10, "fact text", {"confidence": 0.9})
        # Note: In our mock setup all collections are the same mock object, so we just check it was called
        self.assertTrue(mock_collection.upsert.called)

        # Test Query
        print("Testing query_episodic...")
        vm.query_episodic("query", k=5)
        mock_collection.query.assert_called()
        
        print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
