import unittest
from unittest.mock import MagicMock, patch, call
import json
import os
import sys

# Ensure we can import the module
sys.path.append(os.getcwd())
import research_agent

class TestCoveragePersistence(unittest.TestCase):

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_coverage_flow(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Subquestion Coverage Flow ---")
        
        # Setup Env
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Context
            mock_router.return_value = {'episodic': {'ids': []}, 'semantic': {'ids': []}}
            
            # 2. Mock Coverage Lookups
            # Q1 is covered, Q2 is not
            def get_cov_side_effect(topic, q):
                if q == "Q1":
                    return {'created_at': '2025-01-01', 'topic': topic, 'subquestion': q}
                return None
            mock_truth.get_coverage.side_effect = get_cov_side_effect
            
            # 3. LLM Responses
            mock_client = mock_openai.return_value
            
            # Subquestions
            r1 = MagicMock()
            r1.choices[0].message.content = json.dumps({"questions": ["Q1", "Q2"]})
            
            # Decision Gate (Only receives Q2!)
            # Q2 is missing -> Needs Web
            r2 = MagicMock()
            r2.choices[0].message.content = json.dumps({
                "needs_web": True,
                "web_needed_for": ["Q2"],
                "subquestion_statuses": [{"question": "Q2", "status": "missing"}] # Q1 not here
            })
            
            # Summary (for Q2 web search)
            r3 = MagicMock()
            r3.choices[0].message.content = "Summary Q2"
            
            # Facts
            r4 = MagicMock()
            r4.choices[0].message.content = json.dumps({"facts": []})
            
            mock_client.chat.completions.create.side_effect = [r1, r2, r3, r4]
            
            # 4. Web Search Mocks
            mock_web.return_value = [{'link': 'http://test.com'}]
            mock_fetch.return_value = {'text': 'Content Q2', 'title': 'Page'}
            
            # 5. Run
            trace = research_agent.run_research("Test Topic")
            
            # 6. Assertions
            
            # A. Verify Q1 was skipped in LLM eval
            # Check the second call to OpenAI (the decision gate)
            call_args = mock_client.chat.completions.create.call_args_list
            eval_call = call_args[1] 
            prompt = eval_call[1]['messages'][0]['content'] # or kwargs
            self.assertIn('Q2', prompt)
            self.assertNotIn('Q1', prompt)
            print("PASS: Q1 was excluded from LLM evaluation prompt.")
            
            # B. Verify Q2 triggered web search
            mock_web.assert_called_with("Q2", num_results=3)
            
            # C. Verify Persistence
            # Expect add_coverage call for Q2 (Web result)
            # episode id logic: mock_truth.add_episode returns a mock object or int
            # We need to see if add_coverage was called for Q2
            
            # Filter calls to add_coverage
            add_cov_calls = [c for c in mock_truth.add_coverage.call_args_list]
            self.assertTrue(len(add_cov_calls) > 0)
            
            # Verify Q2 coverage saved
            q2_saved = False
            for args, kwargs in add_cov_calls:
                if args[1] == "Q2":
                    q2_saved = True
            
            self.assertTrue(q2_saved)
            print("PASS: Q2 coverage was saved after web search.")

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_trace_consistency(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Trace Consistency ---")
        
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Setup minimal mocks
            mock_router.return_value = {'episodic': {'ids': []}, 'semantic': {'ids': []}}
            mock_truth.get_coverage.return_value = None # No existing coverage
            
            # 2. LLM Responses
            mock_client = mock_openai.return_value
            # Subquestions
            r1 = MagicMock()
            r1.choices[0].message.content = json.dumps({"questions": ["Q_Web"]})
            # Decision (Missing)
            r2 = MagicMock()
            r2.choices[0].message.content = json.dumps({
                "needs_web": True,
                "web_needed_for": ["Q_Web"],
                "subquestion_statuses": [{"question": "Q_Web", "status": "missing"}]
            })
            # Summary + Facts
            r3 = MagicMock()
            r3.choices[0].message.content = "Summary"
            r4 = MagicMock()
            r4.choices[0].message.content = json.dumps({"facts": []})
            mock_client.chat.completions.create.side_effect = [r1, r2, r3, r4]
            
            # 3. Web Search (Successful)
            mock_web.return_value = [{'link': 'http://foo.com'}]
            mock_fetch.return_value = {'text': 'content', 'title': 'title'}
            
            # 4. Run
            trace = research_agent.run_research("Topic")
            
            # 5. Assertions
            # Find status for Q_Web
            status_entry = next((s for s in trace["subquestion_statuses"] if s["question"] == "Q_Web"), None)
            self.assertIsNotNone(status_entry)
            self.assertEqual(status_entry["status"], "satisfied")
            self.assertIn("Answered via web", status_entry["rationale"])
            
            print("PASS: Trace status was updated to satisfied after web search.")

            print("PASS: Trace status was updated to satisfied after web search.")

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    def test_normalized_coverage(self, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Normalized Coverage ---")
        
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            # 1. Setup Logic
            mock_router.return_value = {'episodic': {'ids': []}, 'semantic': {'ids': []}}
            
            # Mock get_coverage to return None (Exact match fails)
            mock_truth.get_coverage.return_value = None
            
            # Mock get_coverage_by_topic to return a normalized match
            # "what are the trends" ~= "key trends"
            mock_truth.get_coverage_by_topic.return_value = [
                {
                    'subquestion': 'What are the key trends?',
                    'normalized_subquestion': 'trends', 
                    'created_at': '2025-01-01',
                    'topic': 'Topic'
                }
            ]
            
            # 2. LLM Responses (Should NOT be called for covered Q)
            mock_client = mock_openai.return_value
            r1 = MagicMock()
            r1.choices[0].message.content = json.dumps({"questions": ["What are the latest trends?"]})
            # If Q is covered, needs_web should be false (if only 1 Q)
            # But wait, run_research recomputes needs_web.
            # If covered pre-check works, we skip eval.
            mock_client.chat.completions.create.side_effect = [r1] 
            
            # 3. Run
            trace = research_agent.run_research("Topic")
            
            # 4. Assertions
            # Status should be satisfied
            status = trace["subquestion_statuses"][0]
            self.assertEqual(status["question"], "What are the latest trends?")
            self.assertEqual(status["status"], "satisfied")
            self.assertIn("Previously covered", status["rationale"])
            
            # Verify normalization logic was used
            # "what are the latest trends" -> "trends"
            # "key trends" -> "trends"
            # Jaccard("trends", "trends") = 1.0 > 0.8
            
            print("PASS: Fuzzy matched subquestion.")

if __name__ == '__main__':
    unittest.main()
