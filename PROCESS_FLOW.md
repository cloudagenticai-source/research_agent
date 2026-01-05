# Research Agent Process Flow

This document details the holistic execution flow of the `run_research` function, highlighting interactions with the LLM, Database (SQLite), Vector Store (Chroma), and Web.

## Phase 1: Initialization & Context (The "Brain" Check)
**Goal:** Setup the environment and recall what the agent already knows.

1.  **Runtime Setup** (`_init_runtime`)
    *   **SQLite**: Connects to `data/memory.db`.
    *   **Chroma**: Connects to the vector store (`./chroma_db`).
    *   **Session**: Generates a unique UUID for the run.

2.  **Retrieve Context** (`_retrieve_context`)
    *   **Chroma Call**: Sends `topic` to 3 vector collections: `episodic` (Web pages), `semantic` (Facts), `procedural` (Skills).
    *   **Result**: Returns a list of relevant Memory IDs.

3.  **Skill & Policy Selection** (`_select_skill_and_policy`)
    *   **Logic**: Finds a relevant skill (e.g., "Research Report") from context.
    *   **File Read**: Loads `skills.yaml` to set the **Execution Policy** (e.g., `freshness_days=180`, `allow_web=True`).

## Phase 2: Planning (The "Strategy")
**Goal:** Break the topic down into actionable steps.

4.  **Generate Sub-questions** (`_generate_subquestions`)
    *   **LLM Call** (GPT-4o): "Break [Topic] into 4-6 specific sub-questions."
    *   **Output**: A list of strings (e.g., `["What are the trends?", "Is it productive?"]`).
    *   *Note: This is pure brainstorming; it does not check memory.*

## Phase 3: The Decision Gate (The "Filter")
**Goal:** Determine if we can skip web search by reusing memory.

5.  **Coverage Check** (`_decision_gate`)
    *   **SQLite Call**: `get_coverage_by_topic(topic)` fetches all past successful answers.
    *   **Logic Loop** (For each new question):
        1.  **Exact Match**: Is this question string exactly in the DB?
        2.  **Fuzzy Match**: Calculates **Jaccard Similarity**. If > 0.8, it counts as a match.
        3.  **Freshness Check**: Checks `created_at`. If older than 180 days (policy), discard it.
    *   **Result**: Questions are marked as `satisfied` (have memory coverage) or `needs_web`.

## Phase 4: Execution (The "Action")
**Goal:** Fill in the gaps.

6.  **Persist "Memory Wins"** (`_persist_memory_coverage`)
    *   **SQLite Write**: Saves a record linking the *current question* to the *old evidence IDs*. This reinforces the memory link.

7.  **Web Search & Ingestion** (If `needs_web` is True)
    *   **Web Call** (SerpAPI): Searches for the missing questions.
    *   **Web Call** (Requests): Downloads HTML for top results.
    *   **LLM Call**: "Extract key facts from this text." (Runs for each page).
    *   **Persistence**:
        *   **SQLite Write**: `add_episode` (Notes) and `add_fact` (Facts).
        *   **Chroma Write**: `upsert_episode` and `upsert_fact` (Embeddings).
    *   **Result**: New IDs are generated (e.g., Episode 50, 51).

8.  **Persist "Web Wins"** (`_persist_web_coverage_and_update_statuses`)
    *   **SQLite Write**: Saves a record linking the *Web Questions* to the *New IDs*.
    *   **Logic**: Ensures future runs can reuse this new data immediately.

## Phase 5: Reporting (The "Summary")
**Goal:** Synthesize the findings for the user.

9.  **Compress Summaries** (`_attach_compressed_summaries`)
    *   **SQLite Call**: Fetches full text for all relevant IDs (Episodes 50, 51).
    *   **LLM Call**: "Summarize the answer to [Question] using ONLY this evidence."
    *   **Output**: A concise summary is attached to the trace (Ram Only).

10. **Final Return**
    *   Returns the `trace` dictionary for CLI display or Report Writing.
