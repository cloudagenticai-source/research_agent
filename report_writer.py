import os
import re
from openai import OpenAI
import memory_vector
import memory_truth
import router

def generate_report(topic: str, max_episodes: int = 5, max_facts: int = 15, session_id: str = None) -> str:
    """
    Generate a cited research report based on stored memories.
    
    Args:
        topic: The research topic.
        max_episodes: Maximum number of episodic memories to include in context.
        max_facts: Maximum number of semantic facts to include in context.
        
    Returns:
        str: The generated report text.
    """
    
    # 1. Initialize
    memory_truth.init_db()
    vm = memory_vector.VectorMemory()
    openai_client = OpenAI()
    
    # 2. Retrieve Context
    print(f"Retrieving memory context for report on: {topic}")
    context = router.retrieve_router(vm, topic, k_epi=max_episodes, k_sem=max_facts)
    
    # 3. Strict Context Assembly (Source of Truth: SQLite)
    # Determine Session ID
    if not session_id:
        session_id = memory_truth.get_latest_session_id(topic)
        print(f"No session specified. Using latest session: {session_id}")
    
    if not session_id:
        return "No session found for this topic."

    # Fetch ALL episodes for this topic AND session from Truth Store
    topic_episodes = memory_truth.get_episodes_by_topic_and_session(topic, session_id)
    
    # 4. Ranking & Selection
    # Map Chroma IDs to integers for intersection
    episode_doc_ids = context.get('episodic', {}).get('ids', [])
    chroma_ints = []
    for doc_id in episode_doc_ids:
        try:
            chroma_ints.append(int(doc_id.split(':')[1]))
        except (IndexError, ValueError):
            continue
            
    # Create valid pool from SQLite (Topic Scoped)
    # We construct a map for O(1) lookup
    topic_map = {e['id']: e for e in topic_episodes}
    
    # Selection Strategy:
    # 1. Intersection (Chroma IDs that exist in Topic Map)
    # 2. Fallback (Recent Topic Episodes that were missed by Chroma)
    
    final_episodes = []
    seen_ids = set()
    
    # Add Chroma matches first (maintaining Chroma rank)
    for cid in chroma_ints:
        if cid in topic_map and cid not in seen_ids:
            final_episodes.append(topic_map[cid])
            seen_ids.add(cid)
            
    # Fill quota with recent topic episodes if needed
    if len(final_episodes) < max_episodes:
        for e in topic_episodes:
            if e['id'] not in seen_ids:
                final_episodes.append(e)
                seen_ids.add(e['id'])
                if len(final_episodes) >= max_episodes:
                    break
    
    episodes = final_episodes

    # Load Semantic Facts (Standard Chroma Load, filtered by topic)
    fact_doc_ids = context.get('semantic', {}).get('ids', [])
    fact_ints = []
    for doc_id in fact_doc_ids:
        try:
            fact_ints.append(int(doc_id.split(':')[1]))
        except (IndexError, ValueError):
            continue
    facts = memory_truth.get_facts_by_ids(fact_ints)
    facts = [f for f in facts if f['topic'].lower() == topic.lower()]
    # Also filter facts by session validity? 
    # Since semantic search crosses boundaries, we must ensure facts belong to our session.
    # We can fetch ALL session facts and intersect, or just trust the DB helper if we load from scratch?
    # The current code loads via Chroma IDs -> then checks DB.
    # Better approach for strictness: Use get_facts_by_topic_and_session and intersect with Chroma results.
    
    session_facts = memory_truth.get_facts_by_topic_and_session(topic, session_id)
    session_fact_map = {f['id']: f for f in session_facts}
    
    final_facts = []
    seen_fact_ids = set()
    
    # 1. Chroma Rank Intersection
    for fid in fact_ints:
        if fid in session_fact_map and fid not in seen_fact_ids:
            final_facts.append(session_fact_map[fid])
            seen_fact_ids.add(fid)
            
    # 2. Quota Fill
    if len(final_facts) < max_facts:
        for f in session_facts:
            if f['id'] not in seen_fact_ids:
                final_facts.append(f)
                seen_fact_ids.add(f['id'])
                if len(final_facts) >= max_facts:
                    break
    
    facts = final_facts

    # 5. Validation Check
    # We must have topic-scoped allowed_urls to proceed
    allowed_urls = []
    for e in episodes:
        url = e.get('url')
        if url:
             allowed_urls.append(url)
    
    allowed_urls = sorted(list(set(allowed_urls)))
    
    if not allowed_urls:
         return "No topic-scoped ingested sources available to generate a cited report for this topic."

    # 6. Assemble Context Pack
    
    # Format Skills
    skill_ids = context.get('procedural', {}).get('ids', [])
    selected_skill = skill_ids[0] if skill_ids else "General Research Report"
    skill_context = f"Selected Approach/Skill: {selected_skill}"
    
    # Format Facts
    facts_text = "Key Semantic Facts:\n"
    for f in facts:
        # Note: We do NOT whitelist fact source URLs unless they are also in the evidence episodes.
        # This prevents citing sources we don't have full context for.
        source_url = f.get('source_url')
        source_info = f"(Source: {source_url})" if source_url and source_url in allowed_urls else ""
        facts_text += f"- {f['subject']} {f['predicate']} {f['object']} [Confidence: {f['confidence']}] {source_info}\n"

    # Format Evidence
    evidence_text = "Episodic Evidence (Source Notes):\n"
    for e in episodes:
        url = e.get('url', 'No URL')
        title = e.get('title', 'Untitled')
        
        # Format notes
        raw_notes = e.get('notes', '')
        note_bullets = [n.strip() for n in raw_notes.replace('\n', ' ').split('. ') if n.strip()]
        formatted_notes = "\n".join(f"    * {n}" for n in note_bullets)
        
        evidence_text += f"- Title: {title}\n  URL: {url}\n  Evidence Notes:\n{formatted_notes}\n\n"
    
    allowed_refs_text = "Allowed References:\n" + "\n".join(f"- {u}" for u in allowed_urls)

    system_prompt = (
        "You are an advanced research assistant. Write a comprehensive, well-structured report based ONLY on the provided context.\n"
        "You may ONLY cite URLs from the Allowed References list provided below.\n"
        "Do not invent information or citations not present in the context.\n"
        "Prioritize evidence and facts over narrative fluff."
    )

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"{skill_context}\n\n"
        f"{allowed_refs_text}\n\n"
        f"{facts_text}\n\n"
        f"{evidence_text}\n\n"
        "Instructions:\n"
        "1. Write an Executive Summary.\n"
        "2. List Key Findings.\n"
        "3. Discuss the topic in detail, organized by sub-themes.\n"
        "4. Note any Limitations.\n"
        "5. Include a References section listing the URLs used.\n"
        "\nSTRICT CITATION RULES:\n"
        "- Do not introduce any source not explicitly listed in Allowed References.\n"
        "- If you cannot support a claim with an allowed URL, label it as 'Unverified' and do not cite.\n"
        "\nGenerate the report now."
    )

    # 6. Generate with LLM
    print("Generating report with LLM...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        report = response.choices[0].message.content
        
        # Post-processing: Validate Citations
        # Pass 1: Handle Markdown links [text](url)
        # We find all markdown links and check if the URL is allowed.
        md_link_pattern = r'\[([^\]]+)\]\((http[^)]+)\)'
        
        def replace_md_link(match):
            text = match.group(1)
            url = match.group(2)
            if url in allowed_urls:
                return match.group(0) # Keep as is
            else:
                return f"[{text}]([citation unavailable])"
        
        report = re.sub(md_link_pattern, replace_md_link, report)

        # Pass 2: Handle Parenthetical Citations (url)
        # We find parenthetical urls that might have been missed or are standalone
        paren_link_pattern = r'\((http[^)]+)\)'
        
        def replace_paren_link(match):
            url = match.group(1)
            if url in allowed_urls:
                return match.group(0)
            else:
                return "[citation unavailable]"

        report = re.sub(paren_link_pattern, replace_paren_link, report)
        
        # Pass 3: Handle Bare URLs (just http...)
        # Careful not to destroy the ones we just validated/fixed?
        # Actually, if they are valid, they are in allowed_urls.
        # But if we already processed them in Pass 1 or 2, we should be careful.
        # Simple approach: Find all http... sequences. If they are in allowed_urls, great.
        # If not, check if they are part of a structure we approve?
        # A simpler robust way:
        # We already handled [text](url) and (url).
        # Any remaining http... that is NOT in allowed_urls should likely be nuked?
        # But wait, Pass 1 and 2 might have left valid URLs alone.
        # So we can just scan for any http... token.
        # If it's valid, leave it. If it's invalid, redact it.
        # Caveat: We need to avoid redacting the "http" inside "[citation unavailable]" (wait, that doesn't have http).
        
        # Let's do a general pass for any http(s)://... sequence.
        # We need a regex that grabs the whole URL.
        # Note: This might overlap with previous passes if we aren't careful, but since we replaced invalid ones
        # with [citation unavailable], the only URLs left should be valid ones OR bare invalid ones.
        
        url_pattern = r'(https?://[^\s\)]+)'
        
        def replace_bare_url(match):
            url = match.group(1)
            # Strip trailing punctuation sometimes caught by greedy regex
            # e.g. "http://foo.com." -> "http://foo.com"
            clean_url = url.rstrip(').,;')
            
            if clean_url in allowed_urls:
                return match.group(0)
            else:
                return "[citation unavailable]"
                
        report = re.sub(url_pattern, replace_bare_url, report)

        # Count redactions for the warning note
        replaced_count = report.count("[citation unavailable]")
        
        # Inject Limitation note if replacements happened
        if replaced_count > 0:
            limitations_header = "## Limitations"
            warning = "\n> **Note:** Some claims in this report could not be linked to a verified source from the provided context and have been marked as [citation unavailable].\n"
            if limitations_header in report:
                report = report.replace(limitations_header, limitations_header + warning)
            else:
                report += f"\n\n{limitations_header}{warning}"

        # Overwrite References Section
        ref_header = "## References"
        references_content = "\n".join(f"- {u}" for u in allowed_urls)
        
        # Split and reconstruct to ensure clean replacement
        if ref_header in report:
            parts = report.split(ref_header)
            report = parts[0] + f"{ref_header}\n\n{references_content}"
        else:
            report += f"\n\n{ref_header}\n\n{references_content}"

        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    # Smoke test
    print(generate_report("Remote Work"))
