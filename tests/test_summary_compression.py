
import unittest
from unittest.mock import MagicMock, patch
import json
import os
import research_agent

class TestSummaryCompression(unittest.TestCase):
    
    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    @patch('research_agent.memory_builders.load_skills')
    def test_summary_compression(self, mock_load_skills, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Summary Compression ---")
        
        # 1. Setup Skill
        mock_router.return_value = {'procedural': {'ids': []}, 'episodic': {'ids': []}, 'semantic': {'ids': []}}
        mock_load_skills.return_value = []
        
        # Mock LLM calls:
        # 1. Subquestions
        r1 = MagicMock()
        r1.choices[0].message.content = json.dumps({"questions": ["Q1"]})
        
        # 2. Coverage found for Q1
        c1 = {
            'subquestion': 'Q1', 
            'created_at': '2026-01-02', 
            'subquestion': 'Q1', 
            'normalized_subquestion': 'q1',
            'id': 1,
            'episode_ids': '[1]',
            'fact_ids': '[]'
        }
        mock_truth.get_coverage.return_value = c1
        mock_truth.get_coverage_by_topic.return_value = [c1]
        
        # 3. Summary Compression Response (Just the text)
        r_sum = MagicMock()
        r_sum.choices[0].message.content = "The answer is 42."
        
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [r1, r_sum]
        
        # Mock Episodes/Facts for summary
        mock_truth.get_episodes_by_ids.return_value = [{'id': 1, 'created_at': '2025-01-01', 'notes': 'Deep thought.'}]
        mock_truth.get_facts_by_ids.return_value = []

        # 2. Run
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
             trace = research_agent.run_research("Topic")
            
        # 3. Assertions
        self.assertIn("compressed_summaries", trace)
        self.assertEqual(len(trace["compressed_summaries"]), 1)
        # Check dictionary structure
        self.assertEqual(trace["compressed_summaries"]["Q1"]["summary"], "The answer is 42.")
        print("PASS: Summary compression executed.")

if __name__ == '__main__':
    unittest.main()
