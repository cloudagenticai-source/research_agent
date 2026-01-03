
import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime, timedelta
import research_agent

class TestFreshnessPolicy(unittest.TestCase):
    
    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    @patch('research_agent.memory_builders.load_skills')
    def test_freshness_check(self, mock_load_skills, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Freshness Policy ---")
        
        # 1. Setup Skill
        # Policy: 1 day freshness
        mock_router.return_value = {'procedural': {'ids': ['skill_fresh']}}
        mock_load_skills.return_value = [
            {
                'id': 'skill_fresh', 
                'execution_policy': {'freshness_days': 1}
            }
        ]
        
        # Coverage 1: Fresh (Today)
        c1 = {
            'subquestion': 'Q_Fresh',
            'normalized_subquestion': 'fresh',
            'created_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            'topic': 'Topic'
        }
        
        # Coverage 2: Stale (10 days old)
        c2 = {
            'subquestion': 'Q_Stale',
            'normalized_subquestion': 'stale',
            'created_at': (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
            'topic': 'Topic'
        }

        # Setup get_coverage behavior
        def get_coverage_side_effect(topic, q):
            if q == 'Q_Fresh': return c1
            if q == 'Q_Stale': return c2
            return None
        mock_truth.get_coverage.side_effect = get_coverage_side_effect
        mock_truth.get_coverage_by_topic.return_value = [c1, c2]
        
        # Mock LLM Subquestions
        mock_client = mock_openai.return_value
        r1 = MagicMock()
        r1.choices[0].message.content = json.dumps({"questions": ["Q_Fresh", "Q_Stale"]})
        
        # Mock Evaluation for Stale Question
        # Stale -> Uncovered -> Evaluate -> Missing (in Mock) -> Needs Web
        r2 = MagicMock()
        r2.choices[0].message.content = json.dumps({
            "subquestion_statuses": [{"question": "Q_Stale", "status": "missing"}],
            "needs_web": True,
            "web_needed_for": ["Q_Stale"]
        })
        
        mock_client.chat.completions.create.side_effect = [r1, r2]

        # 2. Run
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
             trace = research_agent.run_research("Topic")
            
        # 3. Assertions
        statuses = {s['question']: s['status'] for s in trace['subquestion_statuses']}
        
        # Q_Fresh should be 'satisfied' via Memory (skipped evaluation)
        self.assertEqual(statuses.get('Q_Fresh'), 'satisfied')
        # Check rationale for "Previously covered"
        s_fresh = next(s for s in trace['subquestion_statuses'] if s['question'] == 'Q_Fresh')
        self.assertIn("Previously covered", s_fresh['rationale'])

        # Q_Stale found in memory but marked stale, so evaluate -> missing -> needs web
        # It won't be in statuses YET from pre-check.
        # It will be in 'web_needed_for'
        self.assertIn("Q_Stale", trace['web_needed_for'])
        print("PASS: Stale question triggered web search.")
        print("PASS: Fresh question reused memory.")

if __name__ == '__main__':
    unittest.main()
