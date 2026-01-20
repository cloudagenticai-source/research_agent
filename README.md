# Research Agent: Memory-First Autonomous Intelligence

> **Beyond the Prompt Wrapper**
> Most "AI agents" are just glorified prompt pipelines. I wanted to build something that actually thinks before it acts.

This Research Agent is not just a chatbot with tools. It is an autonomous system built on a **Triple-Memory Architecture** where long-term memory is the core driver of intelligence, not an afterthought. By indexing every research session into a structured "brain," the agent becomes smarter over time—slashing redundant API calls by 40% and eliminating hallucinated citations.

![Research Agent Architecture](./Research%20Agent.png)

## 🧠 The Architecture: A Triple-Memory System

Instead of a single ephemeral context window, the agent utilizes a **Dual-Store Memory** system combining **ChromaDB** (Vector) and **SQLite** (Relational) to manage three distinct types of memory:

### 1. Episodic Memory (The "Journal")
*   **Storage**: SQLite (`episodes` table)
*   **Function**: Tracks every specific research session, sub-query, and source URL.
*   **Benefit**: If the agent researched "Quantum Computing trends" last week, it recalls the exact source, date, and outcome without needing to re-search.

### 2. Semantic Memory (The "Library")
*   **Storage**: ChromaDB (Vector Index) + SQLite (`facts` table)
*   **Function**: Stores atomic facts extracted from episodes (e.g., *Subject: GPT-4, Predicate: released_by, Object: OpenAI*).
*   **Benefit**: Uses vector similarity search to fuzzy-match new, unstructured questions against a massive database of prior knowledge.

### 3. Procedural Memory (The "Playbook")
*   **Storage**: `skills/skills.yaml`
*   **Function**: Encodes deterministic strategies and execution policies (e.g., "for Medical topics, prioritize PubMed").
*   **Benefit**: Intelligence scales with design, not just prompt engineering. The agent follows a strict, inspectable process rather than "guessing" the next step.

---

## 🛡️ The Agentic "Decision Gate"

The most powerful component is the **Decision Gate** (`evaluate_subquestions_against_memory`). Before making any external API calls, the agent inspects its own brain.

### How It Works:
1.  **Recall**: The agent retrieves relevant Episodes and Facts for the current topic.
2.  **Evaluate**: An LLM judge evaluates if the existing memory is sufficient to answer the specific sub-questions.
3.  **The 180-Day Rule**: The system enforces strict freshness. Even if data exists, if the `created_at` timestamp is older than **180 days**, it is marked `stale` and a fresh web search is triggered.
4.  **Action**:
    *   **Memory Hit**: Generates answer instantly (0 latency, $0 cost).
    *   **Memory Miss/Stale**: Triggers SerpAPI + BS4 web scraper.

> **Result**: A deterministic, transparent system where you can literally "open the hood" to see why a decision was made.

---

## 🛠️ Tech Stack

*   **Frontend**: Vite + React (TypeScript)
*   **Backend API**: FastAPI (Python)
*   **Intelligence**: GPT-4o
*   **Vector Store**: ChromaDB
*   **Relational Store**: SQLite
*   **Search**: SerpAPI

---

## 🔒 Access

This repository is currently private. For access or technical deep-dives, please contact the owner via LinkedIn DM.
