import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Ensure we can import the module
sys.path.append(os.getcwd())
import research_agent

class TestFactIngestion(unittest.TestCase):

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_fact_ingestion_handles_nulls(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Fact Ingestion with NULL values ---")
        
        # Setup to force web search and ingestion
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Context: Empty
            mock_router.return_value = {'episodic': {'ids': []}, 'semantic': {'ids': []}}
            mock_truth.get_episodes_by_ids.return_value = []
            
            # 2. LLM Responses
            mock_client = mock_openai.return_value
            
            # Subquestions
            r1 = MagicMock()
            r1.choices[0].message.content = json.dumps({"questions": ["Q1"]})
            
            # Decision (NEEDS WEB)
            r2 = MagicMock()
            r2.choices[0].message.content = json.dumps({
                "needs_web": True,
                "web_needed_for": ["Q1"],
                "subquestion_statuses": [] # optional
            })
            
            # Summary
            r3 = MagicMock()
            r3.choices[0].message.content = "Summary text"
            
            # Facts with NULL/Empty values (The Reproduction Case)
            bad_facts = {
                "facts": [
                    {"subject": "S1", "predicate": "P1", "object": None, "confidence": 0.9}, # Null object
                    {"subject": None, "predicate": "P2", "object": "O2", "confidence": None}, # Null subj, conf
                    {"subject": "", "predicate": "", "object": "", "confidence": 0.5}, # Empty strings
                ]
            }
            r4 = MagicMock()
            r4.choices[0].message.content = json.dumps(bad_facts)
            
            mock_client.chat.completions.create.side_effect = [r1, r2, r3, r4]
            
            # 3. Web Search Mocks
            mock_web.return_value = [{'link': 'http://test.com'}]
            mock_fetch.return_value = {'text': 'Some content', 'title': 'Page'}
            
            # 4. Run
            trace = research_agent.run_research("Test Topic")
            
            # 5. Assertions
            # Inspect calls to add_fact
            calls = mock_truth.add_fact.call_args_list
            self.assertEqual(len(calls), 3)
            
            # Check Call 1: Null object -> 'unknown'
            args1, kwargs1 = calls[0]
            self.assertEqual(kwargs1['object_'], 'unknown')
            self.assertEqual(kwargs1['subject'], 'S1')
            
            # Check Call 2: Null subject -> 'unknown', Null conf -> 0.5
            args2, kwargs2 = calls[1]
            self.assertEqual(kwargs2['subject'], 'unknown')
            self.assertEqual(kwargs2['confidence'], 0.5)
            
            # Check Call 3: Empty string -> 'unknown'
            args3, kwargs3 = calls[2]
            self.assertEqual(kwargs3['subject'], 'unknown')
            self.assertEqual(kwargs3['predicate'], 'related to') # Fallback
            self.assertEqual(kwargs3['object_'], 'unknown')

            print("PASS: Fact ingestion handled all NULL/Empty values correctly.")

if __name__ == '__main__':
    unittest.main()
