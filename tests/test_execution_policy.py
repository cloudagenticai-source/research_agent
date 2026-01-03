
import unittest
from unittest.mock import MagicMock, patch
import json
import os
import research_agent

class TestExecutionPolicy(unittest.TestCase):
    
    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    @patch('research_agent.memory_builders.load_skills')
    def test_policy_enforcement_web_disabled(self, mock_load_skills, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Policy: allow_web=False ---")
        
        # 1. Setup Skill with allow_web=False
        mock_router.return_value = {
            'procedural': {'ids': ['skill_no_web']},
            'episodic': {'ids': []}, 
            'semantic': {'ids': []}
        }
        
        mock_load_skills.return_value = [
            {
                'id': 'skill_no_web',
                'name': 'No Web Skill',
                'execution_policy': {
                    'allow_web': False,
                    'reuse_memory': True,
                    'max_sources': 1
                }
            }
        ]
        
        # Mock LLM to return a subquestion that needs web
        mock_client = mock_openai.return_value
        r1 = MagicMock()
        r1.choices[0].message.content = json.dumps({"questions": ["Uncovered Question"]})
        mock_client.chat.completions.create.return_value = r1

        # Mock coverage to return None (so it is uncovered)
        mock_truth.get_coverage.return_value = None
        mock_truth.get_coverage_by_topic.return_value = []
        
        # Mock evaluation to say "missing" (so needs_web WOULD be True)
        # Using return value for evaluate_... call? 
        # Wait, evaluate_subquestions_against_memory calls openai.
        # Ideally we mock evaluate_subquestions_against_memory directly?
        # But we are testing research_agent.py where it is defined.
        # Let's mock the LLM response for evaluation too.
        r2 = MagicMock()
        r2.choices[0].message.content = json.dumps({
            "subquestion_statuses": [{"question": "Uncovered Question", "status": "missing"}],
            "needs_web": True,
            "web_needed_for": ["Uncovered Question"]
        })
        # Sequence of calls: 
        # 1. Subquestions
        # 2. Evaluation
        mock_client.chat.completions.create.side_effect = [r1, r2]

        # 2. Run
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            trace = research_agent.run_research("Topic")
            
        # 3. Assertions
        # Policy should override needs_web
        print(f"Decision: Needs Web? {trace['needs_web']}")
        self.assertFalse(trace['needs_web'])
        self.assertFalse(trace['needs_web'])
        # Check active policy
        self.assertEqual(trace['execution_policy']['allow_web'], False)
        pass

    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    @patch('research_agent.memory_truth')
    @patch('research_agent.memory_builders.load_skills')
    def test_policy_no_reuse_memory(self, mock_load_skills, mock_truth, mock_fetch, mock_web, mock_vm, mock_router, mock_openai):
        print("\n--- Testing Policy: reuse_memory=False ---")
        
        # 1. Setup Skill
        mock_router.return_value = {'procedural': {'ids': ['skill_fresh']}}
        mock_load_skills.return_value = [
            {
                'id': 'skill_fresh', 
                'execution_policy': {'reuse_memory': False, 'allow_web': True}
            }
        ]
        
        # Mock LLM
        mock_client = mock_openai.return_value
        r1 = MagicMock()
        r1.choices[0].message.content = json.dumps({"questions": ["Q1"]})
        
        # Even if coverage exists, it should be ignored!
        # Mock coverage (if called) to return something.
        # But we expect it NOT to be called or at least ignored.
        mock_truth.get_coverage.side_effect = Exception("Should not call get_coverage!")
        
        # Mock Evaluation response
        r2 = MagicMock()
        r2.choices[0].message.content = json.dumps({
            "subquestion_statuses": [{"question": "Q1", "status": "missing"}],
            "needs_web": True
        })
        
        mock_client.chat.completions.create.side_effect = [r1, r2]
        
        # 2. Run
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}):
            try:
                trace = research_agent.run_research("Topic")
                print("PASS: Ran without hitting get_coverage exception (or logic prevented it).")
            except Exception as e:
                # If it raises exception, that means it called get_coverage
                self.fail(f"Called get_coverage when reuse_memory=False: {e}")
                
        self.assertEqual(trace['execution_policy']['reuse_memory'], False)

if __name__ == '__main__':
    unittest.main()
