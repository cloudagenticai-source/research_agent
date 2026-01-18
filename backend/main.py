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
import report_writer

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

async def event_generator(topic: str, mode: str = "research") -> AsyncGenerator[str, None]:
    """Generates SSE events from research agent execution."""
    if not topic.strip():
        yield format_sse(json.dumps({"message": "Topic cannot be empty"}), "error")
        return

    # Queue for thread communication
    q = queue.Queue()
    
    def on_event(msg: str):
        # Filter logs for report mode?
        # User said: "report mode UI must NOT show the terminal log window"
        # But we can still send them, frontend just ignores.
        # Or we can just not send them to save bandwidth.
        # User said: "Show a lightweight 'Generating Report...' indicator... (NOT terminal logs)"
        # Sending logs is fine, frontend ignores logs in 'report' mode anyway.
        if mode == "report":
            # Maybe send status updates occasionally?
            pass
        q.put({"type": "log", "line": msg})

    def run_wrapper():
        try:
            # Run the agent
            if mode == "report":
                q.put({"type": "report_start", "topic": topic})
                
                # Direct Report Generation (Bypassing run_research to avoid new session)
                research_agent.memory_truth.init_db()
                session_id = research_agent.memory_truth.get_latest_session_id(topic)
                
                # Construct minimal trace for frontend compatibility
                trace = {
                    "topic": topic,
                    "session_id": session_id,
                    "subquestions": [],
                    "sources_used": [],
                    "episode_ids": [],
                    "fact_ids": [],
                    "decision_gate_used": False,
                    "reused_memory": True,
                    "web_calls_skipped": True,
                    "needs_web": False,
                    "memory_stats": {
                        "total_questions": 0,
                        "covered_count": 0,
                        "percent": 0.0
                    }
                }
            else:
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

        elif event_type == "report_start":
             yield format_sse(json.dumps({"message": "Generating Report..."}), "report_start")
             
        elif event_type == "error":
             yield format_sse(json.dumps({"message": item["message"]}), "error")
             
        elif event_type == "done":
             trace = item["trace"]
             
             # Compute Final Summary
             final_summary = ""
             
             if mode == "report":
                 # call report_writer
                 try:
                     session_id = trace.get("session_id")
                     if session_id:
                         final_summary = report_writer.generate_report(topic, session_id=session_id)
                     else:
                         final_summary = "Error: No session ID available to generate report."
                 except Exception as e:
                     final_summary = f"Error generating report: {str(e)}"
             else:
                 # Standard Research Summary Logic
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
             
             if mode == "report":
                 yield format_sse(json.dumps(payload), "report_complete")
             else:
                 yield format_sse(json.dumps(payload), "done")

@app.get("/api/research/stream")
async def stream_research(topic: str):
    return StreamingResponse(event_generator(topic, mode="research"), media_type="text/event-stream")

@app.get("/api/report/stream")
async def stream_report(topic: str):
    return StreamingResponse(event_generator(topic, mode="report"), media_type="text/event-stream")
