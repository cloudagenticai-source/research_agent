
import os
import memory_truth

def test_memory_truth():
    # Helper to clean up
    if os.path.exists(memory_truth.DB_PATH):
        os.remove(memory_truth.DB_PATH)

    print("Initializing DB...")
    memory_truth.init_db()
    
    print("Adding episode...")
    ep_id = memory_truth.add_episode(
        topic="test_topic",
        notes="test_notes",
        url="http://example.com",
        title="Test Title"
    )
    print(f"Episode added with ID: {ep_id}")
    
    print("Retrieving episode...")
    ep = memory_truth.get_episode(ep_id)
    print(f"Retrieved episode: {ep}")
    assert ep['topic'] == 'test_topic'
    
    print("Adding fact...")
    fact_id = memory_truth.add_fact(
        topic="test_fact_topic",
        subject="Python",
        predicate="is",
        object_="awesome",
        source_episode_id=ep_id
    )
    print(f"Fact added with ID: {fact_id}")
    
    print("Retrieving fact...")
    fact = memory_truth.get_fact(fact_id)
    print(f"Retrieved fact: {fact}")
    assert fact['subject'] == 'Python'

    print("Listing recent episodes...")
    recent = memory_truth.list_recent_episodes()
    print(f"Recent episodes: {len(recent)}")
    assert len(recent) == 1

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_memory_truth()
