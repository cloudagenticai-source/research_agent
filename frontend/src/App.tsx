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

interface HistoryItem {
    topic: string;
    summary: SummaryData;
    ts: number;
}

function App() {
    const [topic, setTopic] = useState("")
    const [logs, setLogs] = useState<string[]>([])
    const [status, setStatus] = useState<"idle" | "running" | "completed" | "error">("idle")
    const [runMode, setRunMode] = useState<"idle" | "research" | "report">("idle")
    const [summary, setSummary] = useState<SummaryData | null>(null)
    const [history, setHistory] = useState<HistoryItem[]>([])
    const [historyLoaded, setHistoryLoaded] = useState(false)
    const [lastRunType, setLastRunType] = useState<"research" | "report">("research")

    const eventSourceRef = useRef<EventSource | null>(null)
    const logEndRef = useRef<HTMLDivElement>(null)
    const progressTimer = useRef<ReturnType<typeof setInterval> | null>(null)

    const [reportProgress, setReportProgress] = useState(0)

    // Load history once on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem("research_history_v1")
            if (saved) {
                const parsed = JSON.parse(saved)
                if (Array.isArray(parsed)) {
                    setHistory(parsed)
                }
            }
        } catch (e) {
            console.error("Failed to load history", e)
        } finally {
            setHistoryLoaded(true)
        }
    }, [])

    // Save history whenever it changes, but ONLY after initial load
    useEffect(() => {
        if (!historyLoaded) return;
        try {
            localStorage.setItem("research_history_v1", JSON.stringify(history))
        } catch (e) { console.error("Failed to save history", e) }
    }, [history, historyLoaded])

    // Auto-scroll logs
    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [logs])

    const startResearch = (mode: "research" | "report", overrideTopic?: string) => {
        const currentTopic = overrideTopic ?? topic;
        if (!currentTopic.trim()) return;

        // Reset State
        if (overrideTopic) setTopic(overrideTopic);
        setLogs([])
        setSummary(null)
        setStatus("running")
        setRunMode(mode)

        // Progress Logic for Report Mode
        if (mode === "report") {
            setReportProgress(1)
            if (progressTimer.current) clearInterval(progressTimer.current)
            progressTimer.current = setInterval(() => {
                setReportProgress(prev => {
                    if (prev >= 98) return prev
                    return prev + 1
                })
            }, 500)
        } else {
            setReportProgress(0)
        }

        // Close existing
        if (eventSourceRef.current) {
            eventSourceRef.current.close()
        }

        const endpoint = mode === "report" ? "report" : "research"
        const url = `http://localhost:8000/api/${endpoint}/stream?topic=${encodeURIComponent(currentTopic)}`
        const es = new EventSource(url)
        eventSourceRef.current = es

        es.addEventListener("log", (e) => {
            try {
                const data = JSON.parse(e.data)
                setLogs(prev => [...prev, data.line])

                // Check for milestones in report mode
                if (mode === "report" && data.line) {
                    const l = data.line
                    if (l.includes("Initializing report writer")) setReportProgress(5)
                    else if (l.includes("Retrieving memory context")) setReportProgress(15)
                    else if (l.includes("Analyzing and ranking episodic")) setReportProgress(30)
                    else if (l.includes("Analyzing and ranking semantic")) setReportProgress(45)
                    else if (l.includes("Assembling context")) setReportProgress(60)
                    else if (l.includes("Generating report via LLM")) setReportProgress(75)
                    else if (l.includes("Validating citations")) setReportProgress(90)
                    else if (l.includes("Report generation complete")) setReportProgress(98)
                }
            } catch (err) {
                console.error("Error parsing log:", err)
            }
        })

        es.addEventListener("done", (e) => {
            handleCompletion(e, "research")
        })

        es.addEventListener("report_complete", (e) => {
            handleCompletion(e, "report")
        })

        const handleCompletion = (e: MessageEvent, completionMode: "research" | "report") => {
            try {
                const data = JSON.parse(e.data)
                setSummary(data)
                setStatus("completed")
                setLastRunType(completionMode)

                // 100% on completion
                if (completionMode === "report") {
                    setReportProgress(100)
                    if (progressTimer.current) clearInterval(progressTimer.current)
                }

                // Update History (Deduplicate by normalized topic)
                setHistory(prev => {
                    const normTopic = data.topic.trim().toLowerCase()
                    const filtered = prev.filter(h => h.topic.trim().toLowerCase() !== normTopic)
                    const newItem: HistoryItem = { topic: data.topic.trim(), summary: data, ts: Date.now() }
                    return [newItem, ...filtered].slice(0, 50)
                })

            } catch (err) {
                console.error("Error parsing done:", err)
                setLogs(prev => [...prev, "Error parsing result."])
                setStatus("error")
            } finally {
                es.close()
                // Keep runMode active until user interaction or just let it stay?
                // Per requirement: runMode can stay 'idle' or persist. 
                // Let's reset runMode to 'idle' only if we want to enable controls fully, 
                // but usually we want to see the result. 
                // Actually user request says: "runMode stays 'idle' once completed"
                setRunMode("idle")
            }
        }

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

                if (mode === "report") {
                    if (progressTimer.current) clearInterval(progressTimer.current);
                    setReportProgress(0);
                }
            }
        })
    }

    // Cleanup timer on unmount
    useEffect(() => {
        return () => {
            if (progressTimer.current) clearInterval(progressTimer.current)
        }
    }, [])

    const isBusy = status === "running" || runMode === "report"; // Report mode also blocks

    const dateTimeFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit", hour: "numeric", minute: "2-digit" })

    return (
        <div className="app-shell">
            <aside className="sidebar">
                <div className="sidebar-title">Researched Topics</div>
                <div className="topic-list">
                    {history.map(item => (
                        <div
                            key={item.ts}
                            className="history-item"
                            onClick={() => {
                                if (isBusy) return;
                                setTopic(item.topic)
                                if (item.summary) {
                                    setSummary(item.summary)
                                    setStatus("completed")
                                    setLogs([])
                                }
                            }}
                        >
                            <div className="history-left">
                                <div className="history-topic">{item.topic}</div>
                                <div className="history-date">{dateTimeFormat.format(new Date(item.ts))}</div>
                            </div>

                            <button
                                className="run-report-btn"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (isBusy) return;
                                    setTopic(item.topic);
                                    startResearch("report", item.topic);
                                }}
                                disabled={isBusy}
                                aria-label={`Run Report for ${item.topic}`}
                            >
                                <span className="run-report-label">
                                    <span>Run</span>
                                    <span>Report</span>
                                </span>
                            </button>
                        </div>
                    ))}
                    {history.length === 0 && <div style={{ fontSize: '0.8rem', color: '#666', padding: '10px' }}>No history yet.</div>}
                </div>
            </aside>

            <main className="main">
                <header className="main-header">
                    <h1>Research Agent</h1>

                    <div className="card">
                        <input
                            type="text"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                            placeholder="Enter research topic..."
                            disabled={isBusy}
                        />
                        <button onClick={() => startResearch("research")} disabled={isBusy}>
                            {status === "running" ? "Researching..." : "Start Research"}
                        </button>
                    </div>

                    {runMode === "report" && status === "running" && (
                        <div className="report-status-block">
                            <div className="report-status-text">
                                Generating Report… {reportProgress}%
                            </div>
                            <div className="report-progress">
                                <div
                                    className="report-progress-fill"
                                    style={{ width: `${reportProgress}%` }}
                                />
                            </div>
                        </div>
                    )}
                </header>

                <div className="main-scroll">
                    {runMode !== "report" && status === "running" && logs.length > 0 && (
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
                            <h2>{lastRunType === "report" ? "Report Complete" : "Research Complete"}</h2>
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
            </main>
        </div>
    )
}

export default App
