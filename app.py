import os
import sys
import json

# Ensure environment variable is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

import memory_truth
import memory_builders
import memory_vector
import router
import research_agent
import report_writer

def run_smoke_test():
    """Run the initial smoke test."""
    print("--- Initializing Smoke Test ---")
    # 1. Init DB
    memory_truth.init_db()
    print("SQLite DB initialized.")

    # 2. Init Vector Memory
    print("Initializing VectorMemory...")
    vm = memory_vector.VectorMemory()
    print("VectorMemory initialized.")

    # 3. Procedural Memory: Load and Upsert Skills
    print("\n--- Procedural Memory ---")
    skills = memory_builders.load_skills("skills/skills.yaml")
    print(f"Loaded {len(skills)} skills.")
    
    for skill in skills:
        canonical = memory_builders.skill_canonical(skill)
        # Use skill['name'] as part of metadata for display/filtering
        meta = {"name": skill.get("name", "Unknown")}
        vm.upsert_skill(skill['id'], canonical, meta)
        print(f"Upserted skill: {skill['id']}")

    # 4. Episodic Memory: Add Episode and Upsert
    print("\n--- Episodic Memory ---")
    ep_id = memory_truth.add_episode(
        topic="Remote Work Research",
        title="Productivity Trends 2020-2024",
        url="https://example.com/remote-work",
        notes="Analyzed multiple studies showing mixed results depending on hybrid vs fully remote.",
        tags="remote work, productivity, 2024",
        outcome="completed"
    )
    episode = memory_truth.get_episode(ep_id)
    ep_canonical = memory_builders.episode_canonical(episode)
    # Metadata for vector store (can be subset of full DB record)
    ep_meta = {"topic": episode["topic"], "title": episode["title"]}
    vm.upsert_episode(ep_id, ep_canonical, ep_meta)
    print(f"Upserted episode: {ep_id}")

    # 5. Semantic Memory: Add Fact and Upsert
    print("\n--- Semantic Memory ---")
    fact_id = memory_truth.add_fact(
        topic="Remote Work",
        subject="Hybrid work models",
        predicate="correlated with",
        object_="highest employee satisfaction scores",
        confidence=0.85,
        source_episode_id=ep_id
    )
    fact = memory_truth.get_fact(fact_id)
    fact_canonical = memory_builders.fact_canonical(fact)
    fact_meta = {"topic": fact["topic"], "subject": fact["subject"]}
    vm.upsert_fact(fact_id, fact_canonical, fact_meta)
    print(f"Upserted fact: {fact_id}")

    # 6. Retrieval
    query = "Research and report on remote work productivity since 2020 with citations."
    print(f"\n--- Retrieval ---")
    print(f"Query: '{query}'")
    
    results = router.retrieve_router(vm, query)

    def print_category(name, data):
        print(f"\n[{name.upper()}]")
        ids = data.get('ids', [])
        dists = data.get('distances', [])
        metas = data.get('metadatas', [])
        
        if not ids:
            print("  No results found.")
            return

        for i, doc_id in enumerate(ids):
            # Handle cases where lists might not be aligned or empty (though router fixes this, extra safety)
            dist = dists[i] if i < len(dists) else "N/A"
            meta = metas[i] if i < len(metas) else {}
            print(f"  {i+1}. ID: {doc_id} | Dist: {dist} | Meta: {meta}")

    print_category("Procedural", results['procedural'])
    print_category("Semantic", results['semantic'])
    print_category("Episodic", results['episodic'])

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "research":
            if len(sys.argv) < 3:
                print("Usage: python app.py research 'TOPIC'")
                sys.exit(1)
            
            if not os.environ.get("SERPAPI_API_KEY"):
                print("Error: SERPAPI_API_KEY environment variable is not set.")
                print("Please set it to run the research agent.")
                sys.exit(1)

            topic = sys.argv[2]
            try:
                trace = research_agent.run_research(topic, max_sources=5)
                
                print("\n\n=== RESEARCH SUMMARY ===")
                print(f"Topic: {trace['topic']}")
                print(f"Session ID: {trace.get('session_id', 'N/A')}")
                print(f"Skill: {trace['selected_skill']}")
                print(f"Sub-questions: {len(trace['subquestions'])}")
                for q in trace['subquestions']:
                    print(f"  - {q}")
                print(f"Sources Used: {len(trace['sources_used'])}")
                for s in trace['sources_used']:
                    print(f"  - {s}")
                print(f"Episodes Created: {len(trace['episode_ids'])} (IDs: {trace['episode_ids']})")
                print(f"Facts Created: {len(trace['fact_ids'])} (IDs: {trace['fact_ids']})")
                
            except Exception as e:
                print(f"Research failed: {e}")
                import traceback
                traceback.print_exc()
                
        elif command == "report":
            if len(sys.argv) < 3:
                print("Usage: python app.py report 'TOPIC' [--session SESSION_ID]")
                sys.exit(1)
            
            if not os.environ.get("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY environment variable is not set.")
                sys.exit(1)

            topic = sys.argv[2]
            
            # Parse optional flags manually or with argparse (keeping it simple as requested)
            session_id = None
            if "--session" in sys.argv:
                try:
                    idx = sys.argv.index("--session")
                    session_id = sys.argv[idx + 1]
                except IndexError:
                    print("Error: --session flag requires an ID argument.")
                    sys.exit(1)

            try:
                print(f"Generating report for: {topic} (Session: {session_id or 'Latest'})...\n")
                report = report_writer.generate_report(topic, session_id=session_id)
                print("\n\n=== FINAL RESEARCH REPORT ===\n")
                print(report)
                print("\n=============================\n")
            except Exception as e:
                print(f"Report generation failed: {e}")
                import traceback
                traceback.print_exc()

        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python app.py (runs smoke test)")
            print("  python app.py research 'TOPIC'")
            print("  python app.py report 'TOPIC' [--session SESSION_ID]")
    else:
        run_smoke_test()

if __name__ == "__main__":
    main()
