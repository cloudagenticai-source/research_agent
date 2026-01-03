import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Ensure we can import the module
sys.path.append(os.getcwd())
import research_agent
import app

class TestDecisionGate(unittest.TestCase):

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.memory_truth')
    def test_decision_gate_skips_web(self, mock_truth, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Decision Gate: REUSE MEMORY (Recent) ---")
        
        # 1. Setup Mock Router (Returns IDs)
        mock_router.return_value = {
            'episodic': {'ids': [['episode:1', 'episode:2']]},
            'semantic': {'ids': [['fact:10']]}
        }
        
        # 2. Setup Mock DB Lookups
        # Date must be recent (< 180 days from now 2026-01-03)
        # Use 2025-11-01
        mock_truth.get_episodes_by_ids.return_value = [
            {'id': 1, 'created_at': '2025-11-01', 'title': 'Test Ep', 'url': 'http://foo', 'notes': 'Some notes'}
        ]
        mock_truth.get_facts_by_ids.return_value = [
            {'subject': 'S', 'predicate': 'P', 'object': 'O', 'confidence': 0.9}
        ]
        
        # 3. Setup Mock LLM Responses
        mock_client = mock_openai.return_value
        
        # Response 1: Subquestions
        mock_subq_resp = MagicMock()
        mock_subq_resp.choices[0].message.content = json.dumps({"questions": ["Q1", "Q2"]})
        
        # Response 2: Decision Gate (SAYS NO WEB NEEDED)
        mock_decision_resp = MagicMock()
        decision_data = {
            "needs_web": False,
            "web_needed_for": [],
            "subquestion_statuses": [{"question": "Q1", "status": "satisfied", "rationale": "Memory good"}]
        }
        mock_decision_resp.choices[0].message.content = json.dumps(decision_data)
        
        mock_client.chat.completions.create.side_effect = [
            mock_subq_resp, 
            mock_decision_resp
        ]
        
        # 4. Run
        trace = research_agent.run_research("Test Topic")
        
        # 5. assertions
        self.assertTrue(trace['reused_memory'])
        self.assertTrue(trace['web_calls_skipped'])
        mock_web.assert_not_called()
        # Verify DB lookup was called
        mock_truth.get_episodes_by_ids.assert_called()
        print("PASS: Web search was skipped based on detailed memory.")

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_decision_gate_stale_triggers_web(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Decision Gate: STALE MEMORY ---")
        
        # Setup Environment for API Key check
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Context
            mock_router.return_value = {
                'episodic': {'ids': [['episode:1']]},
                'semantic': {'ids': []}
            }
            # OLD DATE (> 180 days)
            mock_truth.get_episodes_by_ids.return_value = [
                {'id': 1, 'created_at': '2024-01-01', 'title': 'Old Ep', 'url': 'http://foo', 'notes': 'Old notes'}
            ]
            mock_truth.get_facts_by_ids.return_value = []
            
            # 2. LLM Responses
            mock_client = mock_openai.return_value
            
            # Subquestions
            r1 = MagicMock()
            r1.choices[0].message.content = json.dumps({"questions": ["Q1"]})
            
            # Decision (NEEDS WEB due to stale)
            r2 = MagicMock()
            r2.choices[0].message.content = json.dumps({
                "needs_web": True,
                "web_needed_for": ["Q1"],
                "subquestion_statuses": [{"question": "Q1", "status": "stale"}]
            })
            
            # Summary (for ingestion)
            r3 = MagicMock()
            r3.choices[0].message.content = "Summary text"
            
            # Facts (for ingestion)
            r4 = MagicMock()
            r4.choices[0].message.content = json.dumps({"facts": []})
            
            mock_client.chat.completions.create.side_effect = [r1, r2, r3, r4]
            
            # 3. Web Search Mocks
            mock_web.return_value = [{'link': 'http://test.com'}]
            mock_fetch.return_value = {'text': 'Some content', 'title': 'Page'}
            
            # 4. Run
            trace = research_agent.run_research("Test Topic")
            
            # 5. Assertions
            self.assertFalse(trace['reused_memory'])
            self.assertFalse(trace['web_calls_skipped'])
            mock_web.assert_called_with("Q1", num_results=3)
            print("PASS: Web search was executed for stale memory.")

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_decision_gate_needs_web(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Decision Gate: NEEDS WEB ---")
        
        # Setup Environment for API Key check
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Context
            mock_router.return_value = {'episodic': {'ids': []}, 'semantic': {'ids': []}}
            mock_truth.get_episodes_by_ids.return_value = []
            mock_truth.get_facts_by_ids.return_value = []
            
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
                "subquestion_statuses": [{"question": "Q1", "status": "missing"}]
            })
            
            # Summary (for ingestion)
            r3 = MagicMock()
            r3.choices[0].message.content = "Summary text"
            
            # Facts (for ingestion)
            r4 = MagicMock()
            r4.choices[0].message.content = json.dumps({"facts": []})
            
            mock_client.chat.completions.create.side_effect = [r1, r2, r3, r4]
            
            # 3. Web Search Mocks
            mock_web.return_value = [{'link': 'http://test.com'}]
            mock_fetch.return_value = {'text': 'Some content', 'title': 'Page'}
            
            # 4. Run
            trace = research_agent.run_research("Test Topic")
            
            # 5. Assertions
            self.assertFalse(trace['reused_memory'])
            self.assertFalse(trace['web_calls_skipped'])
            mock_web.assert_called_with("Q1", num_results=3)
            print("PASS: Web search was executed for missing info.")

if __name__ == '__main__':
    unittest.main()
