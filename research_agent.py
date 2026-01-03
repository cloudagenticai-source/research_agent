import os
import json
import uuid
import traceback
from openai import OpenAI
import memory_truth
import memory_builders
import memory_vector
import router
import web_search
import web_fetch

def run_research(topic: str, max_sources: int = 5) -> dict:
    """
    Orchestrate the research process.
    
    Args:
        topic: Research topic to investigate.
        max_sources: Maximum number of sources to fetch and ingest.
        
    Returns:
        dict: Execution trace including skill, subquestions, sources, and IDs.
    """
    print(f"--- Starting Research on: {topic} ---")

    # 1. Initialize
    memory_truth.init_db()
    vm = memory_vector.VectorMemory()
    openai_client = OpenAI()

    # Generate Session ID
    session_id = str(uuid.uuid4())
    print(f"Research Session ID: {session_id}")

    trace = {
        "topic": topic,
        "session_id": session_id,
        "selected_skill": None,
        "subquestions": [],
        "sources_used": [],
        "episode_ids": [],
        "fact_ids": []
    }

    # 2. Retrieve Context
    print("Retrieving context from memory...")
    context = router.retrieve_router(vm, topic)

    # 3. Planning
    # Select Skill
    skill_ids = context.get('procedural', {}).get('ids', [])
    if skill_ids:
        trace["selected_skill"] = skill_ids[0]
        print(f"Selected Skill: {trace['selected_skill']}")
    else:
        print("No specific skill found, proceeding with general research.")

    # Generate Sub-questions
    print("Generating sub-questions...")
    prompt = f"Topic: {topic}\n\nBased on this topic, generate 3-6 specific sub-questions to guide web research. Return ONLY a JSON object with a single key 'questions' containing a list of strings."
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        trace["subquestions"] = data.get("questions", [])
        
        # Fallback if list is empty
        if not trace["subquestions"]:
             trace["subquestions"] = [f"key facts about {topic}", f"recent developments in {topic}"]

    except Exception as e:
        print(f"Error generating subquestions: {e}")
        # Fallback
        trace["subquestions"] = [f"key facts about {topic}", f"recent developments in {topic}"]

    print(f"Sub-questions: {trace['subquestions']}")

    # 4. Web Search
    unique_urls = set()
    for q in trace["subquestions"]:
        if len(unique_urls) >= max_sources:
            break
        print(f"Searching: {q}")
        try:
            results = web_search.search_web(q, num_results=3)
            for res in results:
                if res.get('link'):
                    unique_urls.add(res['link'])
        except Exception as e:
            print(f"Search failed for '{q}': {e}")
            
    sorted_urls = list(unique_urls)[:max_sources]
    trace["sources_used"] = sorted_urls
    print(f"Found {len(sorted_urls)} sources.")

    # 5. Fetch & Ingest Episodes
    for url in sorted_urls:
        print(f"Fetching: {url}")
        page_data = web_fetch.fetch_page(url)
        
        if not page_data['text']:
            print(f"Skipping {url}: No text content.")
            continue

        # Extract Summary
        try:
            summary_prompt = f"Summarize the following text related to '{topic}'. Focus on key facts. Keep it under 200 words.\n\nText:\n{page_data['text'][:8000]}"
            summary_resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            summary = summary_resp.choices[0].message.content
        except Exception as e:
            print(f"Summarization failed for {url}: {e}")
            summary = "Summary generation failed."

        # Add Episode
        ep_id = memory_truth.add_episode(
            topic=topic,
            title=page_data.get('title') or "Web Source",
            url=url,
            notes=summary,
            outcome="processed",
            tags="research, web_source",
            session_id=session_id
        )
        trace["episode_ids"].append(ep_id)
        
        # Upsert Episode
        ep = memory_truth.get_episode(ep_id)
        canon = memory_builders.episode_canonical(ep)
        vm.upsert_episode(ep_id, canon, {"topic": topic, "source": "web", "session_id": session_id})
        print(f"Ingested episode {ep_id}")

        # 6. Extract & Ingest Facts
        print(f"Extracting facts from episode {ep_id}...")
        fact_prompt = (
            f"Extract 5-12 key semantic facts from the text below as JSON triples.\n"
            f"Return ONLY a JSON object with a single key 'facts' containing a list of objects.\n"
            f"Format: {{'facts': [{{'subject': '...', 'predicate': '...', 'object': '...', 'confidence': 0.0-1.0}}]}}\n"
            f"Text:\n{summary}"
        )
        
        try:
            fact_resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": fact_prompt}],
                response_format={"type": "json_object"}
            )
            content = fact_resp.choices[0].message.content
            data = json.loads(content)
            facts_data = data.get('facts', [])
            
            if not isinstance(facts_data, list):
                facts_data = []

            for f in facts_data:
                fid = memory_truth.add_fact(
                    topic=topic,
                    subject=f.get('subject', 'unknown'),
                    predicate=f.get('predicate', 'related to'),
                    object_=f.get('object', 'unknown'),
                    confidence=f.get('confidence', 0.5),
                    source_episode_id=ep_id,
                    source_url=url,
                    session_id=session_id
                )
                trace["fact_ids"].append(fid)
                
                # Upsert Fact
                db_fact = memory_truth.get_fact(fid)
                f_canon = memory_builders.fact_canonical(db_fact)
                vm.upsert_fact(fid, f_canon, {"topic": topic, "type": "derived_fact", "session_id": session_id})
            print(f"Extracted {len(facts_data)} facts.")
            
        except Exception as e:
            print(f"Fact extraction failed for episode {ep_id}: {e}")

    print("--- Research Completed ---")
    return trace

if __name__ == "__main__":
    # Smoke test if run directly
    print(run_research("Agentic Memory Systems", max_sources=2))
