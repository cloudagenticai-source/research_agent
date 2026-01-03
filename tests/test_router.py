
import unittest
from unittest.mock import MagicMock
import router

class TestRouter(unittest.TestCase):
    def test_router_flow(self):
        # Mock VectorMemory
        mock_vm = MagicMock()
        
        # Mock responses (list of lists format from Chroma)
        mock_vm.query_episodic.return_value = {
            'ids': [['ep1']],
            'documents': [['doc1']],
            'metadatas': [[{'meta': 1}]],
            'distances': [[0.1]]
        }
        mock_vm.query_semantic.return_value = {
            'ids': [], # Empty result
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        # Procedural returns None fields just to be safe (though chroma usually returns empty lists)
        mock_vm.query_procedural.return_value = {
            'ids': [['skill1']],
            'documents': None,
            'metadatas': None,
            'distances': None
        }

        print("Testing retrieve_router...")
        results = router.retrieve_router(mock_vm, "test query")
        
        # Check Episodic (Should have data)
        self.assertEqual(results['episodic']['ids'], ['ep1'])
        self.assertEqual(results['episodic']['documents'], ['doc1'])
        
        # Check Semantic (Should be empty lists)
        self.assertEqual(results['semantic']['ids'], [])
        self.assertEqual(results['semantic']['documents'], [])
        
        # Check Procedural (Should handle None/Empty key fields)
        self.assertEqual(results['procedural']['ids'], ['skill1'])
        self.assertEqual(results['procedural']['documents'], [])

        print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
