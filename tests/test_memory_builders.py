
import os
import yaml
import memory_builders

def test_memory_builders():
    print("Testing episode_canonical...")
    ep = {
        'topic': 'AI Research',
        'title': 'New Paper',
        'url': 'http://arxiv.org',
        'notes': 'Interesting findings',
        'tags': 'ai,ml',
        'outcome': 'read'
    }
    canon = memory_builders.episode_canonical(ep)
    print(f"Episode Canonical:\n{canon}")
    assert "Topic: AI Research" in canon
    assert "Outcome: read" in canon

    print("\nTesting fact_canonical...")
    fact = {
        'topic': 'Python',
        'subject': 'lists',
        'predicate': 'are',
        'object': 'mutable'
    }
    canon = memory_builders.fact_canonical(fact)
    print(f"Fact Canonical:\n{canon}")
    assert "Python Fact: lists are mutable" in canon

    print("\nTesting skill_canonical...")
    skill = {
        'id': 'test_skill',
        'name': 'Test Skill',
        'description': 'A skill for testing',
        'triggers': ['test', 'run'],
        'steps': ['step 1', 'step 2'],
        'guardrails': ['no bugs']
    }
    canon = memory_builders.skill_canonical(skill)
    print(f"Skill Canonical:\n{canon}")
    assert "Skill: Test Skill" in canon
    assert "Triggers: test, run" in canon

    print("\nTesting load_skills...")
    # Create dummy skills file
    os.makedirs('skills', exist_ok=True)
    with open('skills/test_skills.yaml', 'w') as f:
        yaml.dump([skill, {'name': 'Invalid Skill'}], f) # One valid, one invalid (missing id)
    
    loaded = memory_builders.load_skills('skills/test_skills.yaml')
    print(f"Loaded skills: {len(loaded)}")
    assert len(loaded) == 1
    assert loaded[0]['id'] == 'test_skill'
    
    # Cleanup
    os.remove('skills/test_skills.yaml')

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_memory_builders()
