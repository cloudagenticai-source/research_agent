import os
from datetime import datetime, timedelta
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

# Helper for detailed memory evaluation
def evaluate_subquestions_against_memory(openai_client, topic, subquestions, episodic_ids, semantic_ids):
    """
    Evaluate subquestions against detailed memory evidence from SQLite.
    
    Args:
        openai_client: The OpenAI client instance.
        topic: The research topic.
        subquestions: List of generated subquestions.
        episodic_ids: List of router ID strings (e.g., 'episode:12').
        semantic_ids: List of router ID strings (e.g., 'fact:82').
        
    Returns:
        dict: The decision JSON from the LLM.
    """
    
    # 1. Resolve IDs to Full Records
    # Parse IDs
    ep_ints = []
    for num_str in episodic_ids:
        try:
             ep_ints.append(int(num_str.split(":")[1]))
        except: continue
        
    fact_ints = []
    for num_str in semantic_ids:
        try:
            fact_ints.append(int(num_str.split(":")[1]))
        except: continue
            
    # Fetch from SQLite
    # Limit context size
    episodes = memory_truth.get_episodes_by_ids(ep_ints[:10]) 
    facts = memory_truth.get_facts_by_ids(fact_ints[:25])
    
    # 2. Format Evidence
    evidence_text = "--- EVIDENCE FROM MEMORY ---\n"
    
    if episodes:
        evidence_text += "\n[EPISODIC MEMORIES]\n"
        for e in episodes:
            # Deterministic Staleness Check
            created_at_str = e.get('created_at')
            is_stale = False
            if created_at_str:
                try:
                    # Attempt ISO parse first
                    dt = datetime.fromisoformat(str(created_at_str))
                    if datetime.now() - dt > timedelta(days=180):
                        is_stale = True
                except ValueError:
                    # Try other formats
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(str(created_at_str), fmt)
                            if datetime.now() - dt > timedelta(days=180):
                                is_stale = True
                            break # Success
                        except:
                            pass
            
            evidence_text += f"- ID: {e['id']} | Date: {created_at_str or 'Unknown'} | Stale: {str(is_stale).lower()} | URL: {e.get('url')}\n"
            evidence_text += f"  Title: {e.get('title')}\n"
            evidence_text += f"  Notes: {e['notes'][:300]}...\n" # Truncate notes
            
    if facts:
        evidence_text += "\n[SEMANTIC FACTS]\n"
        for f in facts:
            evidence_text += f"- Fact: {f['subject']} {f['predicate']} {f['object']} (Conf: {f['confidence']})\n"
            
    if not episodes and not facts:
        evidence_text += "(No relevant memory found)"

    # 3. Construct Prompt
    eval_prompt = (
        f"Topic: {topic}\n"
        f"Sub-questions: {json.dumps(subquestions)}\n\n"
        f"{evidence_text}\n\n"
        "Instructions:\n"
        "Evaluate if the existing memory is sufficient to answer each sub-question.\n"
        "RULES:\n"
        "1. \"satisfied\": The EVIDENCE contains specific facts that answer the question.\n"
        "2. \"missing\": The evidence is irrelevant or empty.\n"
        "3. \"stale\": The evidence is relevant BUT represents data marked as \"Stale: true\".\n"
        "   - Do NOT guess dates. Trust the \"Stale\" flag.\n"
        "4. \"contradictory\": The evidence contains conflicting facts.\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        "  \"subquestion_statuses\": [\n"
        "    {\"question\": \"...\", \"status\": \"satisfied|missing|stale|contradictory\", \"rationale\": \"...\"}\n"
        "  ],\n"
        "  \"needs_web\": bool (true if ANY status is missing/stale/contradictory),\n"
        "  \"web_needed_for\": [list of questions requiring search]\n"
        "}"
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Evaluation error: {e}")
        # Default fail-safe
        return {
            "needs_web": True,
            "web_needed_for": subquestions,
            "subquestion_statuses": []
        }


# Helper for subquestion normalization
def normalize_question(q: str) -> str:
    """Normalize question string for fuzzy matching."""
    if not q:
        return ""
    # Lowercase
    q = q.lower()
    # Strip punctuation
    import string
    q = q.strip(string.punctuation)
    # Collapse whitespace
    tokens = q.split()
    # Remove common stop-phrases
    stop_phrases = {"what", "are", "is", "how", "does", "do", "the", "latest", "most", "key"}
    filtered = [t for t in tokens if t not in stop_phrases]
    return " ".join(filtered)

def calculate_jaccard_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between two strings."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def compress_summaries(openai_client, topic, subquestion_statuses):
    """
    Generate compressed summaries using ONLY memory.
    """
    results = {}
    print("Compressing summaries for satisfied questions...")

    # Cache topic coverage for fuzzy fallback
    all_coverage = None 

    for status in subquestion_statuses:
        if status.get("status") != "satisfied":
            continue
            
        q = status.get("question")
        
        # 1. Retrieve Coverage
        cov = memory_truth.get_coverage(topic, q)
        
        if not cov:
            # Fallback: Fuzzy Match
            if all_coverage is None:
                 all_coverage = memory_truth.get_coverage_by_topic(topic)
            
            norm_q = normalize_question(q)
            best_score = 0.0
            best_match = None
            for row in all_coverage:
                 cand_norm = row.get('normalized_subquestion')
                 # Handle legacy rows
                 if not cand_norm:
                     cand_norm = normalize_question(row['subquestion'])
                     
                 score = calculate_jaccard_similarity(norm_q, cand_norm)
                 if score > best_score:
                     best_score = score
                     best_match = row
            
            if best_score >= 0.8:
                cov = best_match
        
        if not cov:
            print(f"Skipping summary for '{q}': No coverage record found.")
            continue
            
        # 2. Results
        try:
             ep_ids = json.loads(str(cov['episode_ids']))
             fact_ids = json.loads(str(cov['fact_ids']))
        except:
             ep_ids = []
             fact_ids = []
             
        # Fetch Data
        episodes = memory_truth.get_episodes_by_ids(ep_ids)
        facts = memory_truth.get_facts_by_ids(fact_ids)
        
        # 3. Context
        context = f"Topic: {topic}\nQuestion: {q}\n\n[EVIDENCE]\n"
        for e in episodes:
            context += f"Source ({e.get('created_at')}): {e.get('notes')[:500]}\n"
        for f in facts:
             context += f"Fact: {f['subject']} {f['predicate']} {f['object']} (Conf: {f['confidence']})\n"
             
        # 4. LLM Call
        prompt = (
            "Summarize the answer to the Question using ONLY the provided memory evidence. "
            "If information is missing, state that explicitly.\n"
            "Keep it factual and under 5 sentences.\n\n"
            f"{context}"
        )
        
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            summary_text = resp.choices[0].message.content.strip()
            
            results[q] = {
                "summary": summary_text,
                "episode_ids": ep_ids,
                "fact_ids": fact_ids
            }
        except Exception as e:
            print(f"Summary failed for {q}: {e}")
            
    return results

# ... run_research ...
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

    trace = {
        "topic": topic,
        "session_id": None, # Will be set later
        "selected_skill": None,
        "subquestions": [],
        "sources_used": [],
        "episode_ids": [],
        "fact_ids": [],
        "decision_gate_used": False,
        "reused_memory": False,
        "web_calls_skipped": False
    }

    # 1. Initialize
    memory_truth.init_db()
    vm = memory_vector.VectorMemory()
    openai_client = OpenAI()

    # Generate Session ID
    session_id = str(uuid.uuid4())
    print(f"Research Session ID: {session_id}")

    trace["session_id"] = session_id
    
    # 2. Retrieve Context
    print("Retrieving context from memory...")
    context = router.retrieve_router(vm, topic)

    # 3. Planning
    # 3. Planning & Policy
    active_policy = {
        "freshness_days": 180,
        "allow_web": True,
        "max_sources": max_sources,
        "reuse_memory": True
    }

    # Select Skill
    skill_ids = context.get('procedural', {}).get('ids', [])
    if skill_ids:
        trace["selected_skill"] = skill_ids[0]
        print(f"Selected Skill: {trace['selected_skill']}")
        
        # Load Policy
        try:
            skills_path = os.path.join(os.path.dirname(__file__), 'skills', 'skills.yaml')
            all_skills = memory_builders.load_skills(skills_path)
            for s in all_skills:
                if s['id'] == trace['selected_skill']:
                    if 'execution_policy' in s:
                        active_policy.update(s['execution_policy'])
                    break
        except Exception as e:
            print(f"Warning: Could not load execution policy: {e}")

    else:
        print("No specific skill found, proceeding with general research.")
        
    trace["execution_policy"] = active_policy
    # Apply Overrides
    max_sources = active_policy.get("max_sources", max_sources)
    print(f"Active Execution Policy: {active_policy}")

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
    
    # --- DECISION GATE ---
    print("Evaluating memory for answers (Deep Verification)...")
    
    # Extract IDs from router context
    # Router returns {'ids': [...], 'documents': [...]} per category
    ep_ids = context.get('episodic', {}).get('ids', [])
    sem_ids = context.get('semantic', {}).get('ids', [])
    
    # Note: Chroma might return list of lists if batch querying, but our router handles single query?
    # Actually router.retrieve_router returns standard dict. 
    # Let's ensure we handle flats lists.
    # The default Chroma output for single query is ids=[['id1', 'id2']].
    # Our router implementation flattens this? Let's check router.py.
    # Checking router.py... it returns raw Chroma results. So ids is [[...]].
    # We need to flatten.
    
    flat_ep_ids = [item for sublist in ep_ids for item in sublist] if ep_ids and isinstance(ep_ids[0], list) else ep_ids
    flat_sem_ids = [item for sublist in sem_ids for item in sublist] if sem_ids and isinstance(sem_ids[0], list) else sem_ids

    decision = {
        "needs_web": False,
        "web_needed_for": [],
        "subquestion_statuses": []
    }

    # Pre-Check Coverage
    uncovered_subquestions = []
    
    should_reuse = active_policy.get("reuse_memory", True)
    if not should_reuse:
        print("Policy Override: reuse_memory=False. Skipping coverage check.")
        uncovered_subquestions = list(trace["subquestions"])
    else:
        print("Checking for existing coverage...")
        
        # Cache all coverage for topic for fuzzy matching
        topic_coverage = memory_truth.get_coverage_by_topic(topic)
        
        for q in trace["subquestions"]:
            # 1. Exact Match
            cov = memory_truth.get_coverage(topic, q)
            
            # 2. Fuzzy Match
            if not cov:
                norm_q = normalize_question(q)
                best_score = 0.0
                best_match = None
                
                for row in topic_coverage:
                    norm_row = row.get('normalized_subquestion')
                    # If row matches legacy format w/o norm column, normalize on fly? 
                    # Or just use row['subquestion']
                    cand_norm = norm_row if norm_row else normalize_question(row['subquestion'])
                    
                    score = calculate_jaccard_similarity(norm_q, cand_norm)
                    if score > best_score:
                        best_score = score
                        best_match = row
                
                if best_score >= 0.8:
                    cov = best_match
                    print(f"  - Fuzzy Match ({best_score:.2f}): '{q}' ~= '{cov['subquestion']}'")

            if cov:
                print(f"  - Covered: {q}")
                decision["subquestion_statuses"].append({
                    "question": q,
                    "status": "satisfied",
                    "rationale": f"Previously covered on {cov['created_at']} (matched: {cov['subquestion']})"
                })
            else:
                uncovered_subquestions.append(q)

    # If we have uncovered questions, evaluate them
    if uncovered_subquestions:
        eval_decision = evaluate_subquestions_against_memory(
            openai_client, 
            topic, 
            uncovered_subquestions, 
            flat_ep_ids or [], 
            flat_sem_ids or []
        )
        # Merge decisions
        decision["subquestion_statuses"].extend(eval_decision.get("subquestion_statuses", []))
    
    # RECOMPUTE needs_web strictly based on statuses
    # This prevents the agent from trusting the LLM's 'needs_web' boolean if it contradicts the status list
    missing_qs = []
    for s in decision["subquestion_statuses"]:
        if s.get("status") != "satisfied":
            missing_qs.append(s.get("question"))
    
    decision["needs_web"] = (len(missing_qs) > 0)
    
    # Enforce Policy
    if not active_policy.get("allow_web", True):
        print("Policy Override: allow_web=False. Forcing needs_web=False.")
        decision["needs_web"] = False
    decision["web_needed_for"] = missing_qs
    
    print(f"Decision: Needs Web? {decision.get('needs_web')}")
    trace["decision_gate_used"] = True
    trace["needs_web"] = decision.get("needs_web", True)
    trace["web_needed_for"] = decision.get("web_needed_for", [])
    trace["subquestion_statuses"] = decision.get("subquestion_statuses", [])
    
    # PERSISTENCE: Save coverage for questions satisfied by memory
    for status in trace["subquestion_statuses"]:
        if status.get("status") == "satisfied" and "Previously covered" not in status.get("rationale", ""):
            print(f"Saving coverage for memory-satisfied question: {status.get('question')}")
            
            # Convert Chroma IDs (strings) to Integer IDs for storage
            ep_ints = []
            for num_str in (flat_ep_ids or []):
                try: ep_ints.append(int(num_str.split(":")[1]))
                except: pass
            
            sem_ints = []
            for num_str in (flat_sem_ids or []):
                try: sem_ints.append(int(num_str.split(":")[1]))
                except: pass

            memory_truth.add_coverage(
                topic, 
                status.get("question"), 
                ep_ints[:10],
                sem_ints[:25],
                normalized_subquestion=normalize_question(status.get("question"))
            )

    if not trace["needs_web"]:
        trace["reused_memory"] = True
        trace["web_calls_skipped"] = True
        print(">>> Skipping Web Search: Memory is sufficient. <<<")
        
        # Summary Compression (Memory Only Path)
        try:
            trace["compressed_summaries"] = compress_summaries(
                openai_client,
                topic,
                trace["subquestion_statuses"]
            )
        except Exception as e:
             print(f"Summary compression failed: {e}")
             trace["compressed_summaries"] = {}
             
        return trace
    
    trace["reused_memory"] = False
    trace["web_calls_skipped"] = False
    
    # 4. Web Search (Conditional)
    unique_urls = set()
    
    # Check API Key before searching
    if not os.environ.get("SERPAPI_API_KEY"):
         raise ValueError("SERPAPI_API_KEY is required for web search but is not set.")

    # Only search for needed questions
    search_queue = trace["web_needed_for"]
    print(f"Searching for {len(search_queue)} missing items...")

    for q in search_queue:
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
            f"Format: {{\"facts\": [{{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\", \"confidence\": 0.0-1.0}}]}}\n"
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
                # Normalize data to prevent NOT NULL constraints
                subj = f.get('subject')
                pred = f.get('predicate')
                obj = f.get('object')
                conf = f.get('confidence')

                fid = memory_truth.add_fact(
                    topic=topic,
                    subject=subj if subj else 'unknown',
                    predicate=pred if pred else 'related to',
                    object_=obj if obj else 'unknown',
                    confidence=conf if conf is not None else 0.5,
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

    # PERSISTENCE: Save coverage for questions answered by Web
    # Assumes the new research covers the questions asked
    new_ep_ids = trace["episode_ids"]
    new_fact_ids = trace["fact_ids"]
    if new_ep_ids:
        for q in trace["web_needed_for"]:
            print(f"Saving coverage for web-answered question: {q}")
            memory_truth.add_coverage(
                topic,
                q,
                new_ep_ids,
                new_fact_ids,
                normalized_subquestion=normalize_question(q)
            )
            
            # TRACE CONSISTENCY: Update status to satisfied
            # Find existing status entry or create new one
            found_status = False
            for status in trace["subquestion_statuses"]:
                if status["question"] == q:
                    status["status"] = "satisfied"
                    status["rationale"] = f"Answered via web in this session (episodes: {new_ep_ids})"
                    found_status = True
                    break
            
            if not found_status:
                trace["subquestion_statuses"].append({
                    "question": q,
                    "status": "satisfied",
                    "rationale": f"Answered via web in this session (episodes: {new_ep_ids})"
                })

    # 7. Summary Compression
    try:
        trace["compressed_summaries"] = compress_summaries(
            openai_client,
            topic,
            trace["subquestion_statuses"]
        )
    except Exception as e:
         print(f"Summary compression failed: {e}")
         trace["compressed_summaries"] = {}

    print("--- Research Completed ---")
    return trace

if __name__ == "__main__":
    # Smoke test if run directly
    print(run_research("Agentic Memory Systems", max_sources=2))
