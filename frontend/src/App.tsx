import { useState, useRef, useEffect } from 'react'

interface Trace {
    session_id: string;
    needs_web: boolean;
    sources_used: string[];
    episode_ids: number[];
    fact_ids: number[];
    reused_memory: boolean;
    memory_stats?: {
        total_questions: number;
        covered_count: number;
        percent: number;
    };
}

interface SummaryData {
    topic: string;
    summary: string;
    trace: Trace;
}

function App() {
    const [topic, setTopic] = useState("")
    const [logs, setLogs] = useState<string[]>([])
    const [status, setStatus] = useState<"idle" | "running" | "completed" | "error">("idle")
    const [summary, setSummary] = useState<SummaryData | null>(null)

    const eventSourceRef = useRef<EventSource | null>(null)
    const logEndRef = useRef<HTMLDivElement>(null)

    // Auto-scroll logs
    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [logs])

    const startResearch = () => {
        if (!topic.trim()) return;

        // Reset State
        setLogs([])
        setSummary(null)
        setStatus("running")

        // Close existing
        if (eventSourceRef.current) {
            eventSourceRef.current.close()
        }

        const url = `http://localhost:8000/api/research/stream?topic=${encodeURIComponent(topic)}`
        const es = new EventSource(url)
        eventSourceRef.current = es

        es.addEventListener("log", (e) => {
            const data = JSON.parse(e.data)
            setLogs(prev => [...prev, data.line])
        })

        es.addEventListener("done", (e) => {
            const data = JSON.parse(e.data)
            setSummary(data)
            setStatus("completed")
            es.close()
        })

        es.addEventListener("error", (e) => {
            // SSE error event (often connection closed, or explicit error)
            try {
                const data = JSON.parse((e as MessageEvent).data)
                if (data.message) {
                    setLogs(prev => [...prev, `ERROR: ${data.message}`])
                }
            } catch {
                // Generic connection error from browser
                //  setLogs(prev => [...prev, "Stream connection closed."])
            }

            if (status === "running") {
                // If we didn't get a 'done' event, it might be a real error
                // But 'error' fires on close too, so we just close to be safe.
                // If we really errored, the backend sends a JSON payload first usually.
                es.close()
            }
        })
    }

    return (
        <div className="app-shell">
            <h1>Research Agent</h1>

            <div className="content">
                <div className="card">
                    <input
                        type="text"
                        value={topic}
                        onChange={(e) => setTopic(e.target.value)}
                        placeholder="Enter research topic..."
                        disabled={status === "running"}
                    />
                    <button onClick={startResearch} disabled={status === "running"}>
                        {status === "running" ? "Researching..." : "Start Research"}
                    </button>
                </div>

                {status === "running" && logs.length > 0 && (
                    <div className="thinking">
                        <div className="thinking-title">Thinking…</div>
                        <div className="thinking-lines">
                            {logs.slice(-8).map((line, i) => (
                                <div key={i} className="thinking-line">{line}</div>
                            ))}
                        </div>
                    </div>
                )}

                {status === "completed" && summary && (
                    <div className="summary-panel">
                        <h2>Research Complete</h2>
                        <p><strong>Topic:</strong> {summary.topic}</p>
                        <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                            {summary.summary}
                        </div>
                        <hr style={{ borderColor: '#444', margin: '1.5rem 0' }} />
                        <details>
                            <summary style={{ cursor: 'pointer', color: '#888' }}>View Run Details</summary>
                            <div style={{ marginTop: '1rem', fontSize: '0.9em', color: '#aaa' }}>
                                <p>Session ID: {summary.trace.session_id}</p>
                                <p>
                                    Memory Reused: {
                                        summary.trace.memory_stats && summary.trace.memory_stats.percent > 0
                                            ? (summary.trace.memory_stats.percent === 100 ? "Yes (100%)" : `Partial (${summary.trace.memory_stats.percent}%)`)
                                            : (summary.trace.reused_memory ? "Yes" : "No")
                                    }
                                </p>
                                {summary.trace.memory_stats && (
                                    <p style={{ fontSize: '0.9em', color: '#6ab0f3' }}>
                                        Coverage Details: {summary.trace.memory_stats.covered_count}/{summary.trace.memory_stats.total_questions} questions answered from memory
                                    </p>
                                )}
                                <p>Web Sources: {summary.trace.sources_used.length}</p>
                                <p>New Episodes: {summary.trace.episode_ids.length}</p>
                                <p>New Facts: {summary.trace.fact_ids.length}</p>
                            </div>
                        </details>
                    </div>
                )}
            </div>
        </div>
    )
}

export default App
