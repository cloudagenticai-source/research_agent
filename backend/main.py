import sys
import os
import json
import asyncio
import queue
from datetime import datetime
from threading import Thread
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directory to path so we can import research_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_sse(data: str, event: str = "message") -> str:
    """Format string data as an SSE event."""
    return f"event: {event}\ndata: {data}\n\n"

async def event_generator(topic: str) -> AsyncGenerator[str, None]:
    """Generates SSE events from research agent execution."""
    if not topic.strip():
        yield format_sse(json.dumps({"message": "Topic cannot be empty"}), "error")
        return

    # Queue for thread communication
    q = queue.Queue()
    
    def on_event(msg: str):
        q.put({"type": "log", "line": msg})

    def run_wrapper():
        try:
            # Run the agent
            trace = research_agent.run_research(topic, max_sources=5, on_event=on_event)
            q.put({"type": "done", "trace": trace})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None) # Sentinel

    # Start research in background thread
    t = Thread(target=run_wrapper, daemon=True)
    t.start()

    # Consuming loop
    while True:
        try:
            # Non-blocking check to keep event loop alive
            # Wait for item with small timeout to yield control
            item = q.get_nowait()
        except queue.Empty:
            if not t.is_alive() and q.empty():
                break
            await asyncio.sleep(0.1)
            continue

        if item is None:
            break

        event_type = item["type"]
        
        if event_type == "log":
             yield format_sse(json.dumps({"line": item["line"]}), "log")
             
        elif event_type == "error":
             yield format_sse(json.dumps({"message": item["message"]}), "error")
             
        elif event_type == "done":
             trace = item["trace"]
             
             # Compute Final Summary
             final_summary = ""
             
             # Priority 1: Compressed Summaries
             compressed = trace.get("compressed_summaries", {})
             if compressed:
                 parts = []
                 for question_text, res in compressed.items():
                     summ = res.get("summary", "")
                     parts.append(f"- **{question_text}**: {summ}")
                 
                 final_summary = "\n".join(parts)
                 # Truncate if huge
                 if len(final_summary) > 1000:
                      final_summary = final_summary[:1000] + "...(truncated)"
                      
             # Priority 2: Fallback
             if not final_summary:
                  final_summary = (
                      f"Completed research on '{topic}'.\n"
                      f"Explored {len(trace['subquestions'])} sub-questions.\n"
                      f"Used {len(trace['sources_used'])} web sources.\n"
                      f"Created {len(trace['episode_ids'])} new memory episodes."
                  )

             # Filter trace for frontend
             safe_trace = {
                 "session_id": trace.get("session_id"),
                 "needs_web": trace.get("needs_web"),
                 "sources_used": trace.get("sources_used"),
                 "episode_ids": trace.get("episode_ids"),
                 "fact_ids": trace.get("fact_ids"),
                 "reused_memory": trace.get("reused_memory"),
                 "memory_stats": trace.get("memory_stats")
             }
             
             payload = {
                 "topic": topic,
                 "summary": final_summary,
                 "trace": safe_trace
             }
             yield format_sse(json.dumps(payload), "done")

@app.get("/api/research/stream")
async def stream_research(topic: str):
    return StreamingResponse(event_generator(topic), media_type="text/event-stream")
