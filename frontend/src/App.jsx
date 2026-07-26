import { useState, useCallback, useRef, useEffect } from "react";
import { Zap, Copy, Check, BarChart3, List, Bug, Loader2, ChevronRight } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import "./index.css";

const API_BASE = "http://localhost:5000/api";

const SAMPLE_TEXT = `Artificial intelligence has transformed numerous industries over the past decade. Companies across sectors from healthcare to finance are leveraging AI to automate tasks and gain critical insights that were previously impossible to obtain.

Machine learning algorithms can identify patterns in data that humans might miss. These patterns enable predictive models that power recommendation systems, fraud detection, and medical diagnosis tools used by millions of people daily.

Natural language processing enables computers to understand and generate human language with remarkable accuracy. Large language models like GPT-4 and Claude can perform complex reasoning tasks, write code, and engage in nuanced conversations on virtually any topic.

These models are trained on vast amounts of text data using self-supervised learning techniques. The attention mechanism, introduced in the transformer architecture, revolutionized NLP by enabling models to consider the full context of a sequence simultaneously.

The cost of running large models at scale is significant, making token efficiency a crucial concern for any production deployment. Reducing prompt length while preserving semantic meaning can cut inference costs by 30-60% for high-volume applications.

TokenShrink addresses this challenge using an information-density-based greedy compression algorithm combined with a dynamic redundancy filter to ensure maximum information preservation within a given token budget.`;

function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const call = useCallback(async (endpoint, body) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "API error");
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);
  return { loading, error, call };
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button className="copy-btn" onClick={copy}>
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card" style={{ "--c": color }}>
      <div className="stat-val">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function ReductionBar({ pct }) {
  return (
    <div className="reduction-bar-wrap">
      <div className="reduction-bar-track">
        <div className="reduction-bar-fill" style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className="reduction-bar-label">{pct}% reduced</span>
    </div>
  );
}

function SentenceBreakdown({ allSentences, selectedSentences }) {
  const selSet = new Set(selectedSentences.map(s => s.index));
  const selMap = Object.fromEntries(selectedSentences.map(s => [s.index, s]));
  return (
    <div className="breakdown">
      {allSentences.map(s => {
        const kept = selSet.has(s.index);
        const meta = selMap[s.index];
        return (
          <div key={s.index} className={`sent-row ${kept ? "kept" : "dropped"}`}>
            <div className="sent-header">
              <span className="sent-idx">S{s.index + 1}</span>
              {kept && meta?.is_anchor && <span className="badge b-anchor">ANCHOR</span>}
              {kept && !meta?.is_anchor && <span className="badge b-kept">KEPT</span>}
              {!kept && <span className="badge b-drop">DROPPED</span>}
              <span className="sent-tok">{s.token_count}t</span>
              {kept && meta?.score > 0 && (
                <span className="sent-score">↑{meta.score.toFixed(4)}</span>
              )}
            </div>
            <div className="sent-text">{s.text}</div>
          </div>
        );
      })}
    </div>
  );
}

function DebugLog({ log }) {
  return (
    <div className="debug-log">
      {(log || []).map((entry, i) => (
        <div key={i} className="log-line">
          <span className="log-i">{String(i + 1).padStart(2, "0")}</span>
          <span>{entry}</span>
        </div>
      ))}
    </div>
  );
}

function TokenChart({ original, compressed }) {
  const saved = original - compressed;
  const data = [
    { name: "Original", tokens: original },
    { name: "Compressed", tokens: compressed },
    { name: "Saved", tokens: saved },
  ];
  const colors = ["#f87171", "#4ade80", "#60a5fa"];
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#475569", fontSize: 10, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, fontFamily: "inherit", fontSize: 11 }}
            labelStyle={{ color: "#cbd5e1" }}
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
          />
          <Bar dataKey="tokens" radius={[4, 4, 0, 0]}>
            {data.map((_, i) => <Cell key={i} fill={colors[i]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="chart-legend">
        {data.map((d, i) => (
          <div key={i} className="legend-row">
            <span className="legend-dot" style={{ background: colors[i] }} />
            <span>{d.name}: <strong>{d.tokens.toLocaleString()}</strong></span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [mode, setMode] = useState("ratio");
  const [ratio, setRatio] = useState(0.5);
  const [budget, setBudget] = useState("");
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState("output");
  const [liveTokens, setLiveTokens] = useState(0);
  const { loading, error, call } = useApi();
  const debounce = useRef(null);

  useEffect(() => {
    clearTimeout(debounce.current);
    debounce.current = setTimeout(async () => {
      if (!text.trim()) { setLiveTokens(0); return; }
      try {
        const r = await fetch(`${API_BASE}/tokenize`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const d = await r.json();
        setLiveTokens(d.token_count || 0);
      } catch {
        setLiveTokens(Math.ceil(text.split(/\s+/).length * 1.3));
      }
    }, 400);
  }, [text]);

  const compress = async () => {
    const body = { text, include_debug: true };
    if (mode === "ratio") body.target_ratio = parseFloat(ratio);
    else if (budget) body.max_tokens = parseInt(budget);
    const data = await call("/compress", body);
    if (data) { setResult(data); setTab("output"); }
  };

  const s = result?.stats;

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────── */}
      <header className="hdr">
        <div className="hdr-bg" />
        <div className="hdr-inner">
          <div className="brand">
            <div className="brand-icon"><Zap size={20} strokeWidth={2.5} /></div>
            <div>
              <h1 className="brand-name">Token<span>Shrink</span></h1>
              <p className="brand-sub">Intelligent Prompt Compression · Greedy Search + TF-IDF</p>
            </div>
          </div>
          <div className="hdr-pills">
            <span className="hdr-pill">Information Density Heuristic</span>
            <span className="hdr-pill">Dynamic Redundancy Filter</span>
            <span className="hdr-pill">Structure Preservation</span>
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Left: Input ─────────────────────────────── */}
        <section className="pane pane-left">
          <div className="pane-title">
            <span>Input Prompt</span>
            <div className="token-live">
              <span className="token-n">{liveTokens.toLocaleString()}</span>
              <span className="token-l">tokens</span>
            </div>
          </div>

          <textarea
            className="textarea"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste your long prompt here…"
            spellCheck={false}
          />

          {/* Mode selector */}
          <div className="mode-row">
            <button className={`mode-btn ${mode === "ratio" ? "active" : ""}`} onClick={() => setMode("ratio")}>
              Compression Ratio
            </button>
            <button className={`mode-btn ${mode === "budget" ? "active" : ""}`} onClick={() => setMode("budget")}>
              Token Budget
            </button>
          </div>

          {mode === "ratio" ? (
            <div className="ratio-block">
              <div className="ratio-info">
                Keep <strong>{Math.round(ratio * 100)}%</strong> of tokens
                {liveTokens > 0 && <span className="ratio-est"> → ~{Math.round(liveTokens * ratio)} output tokens</span>}
              </div>
              <input
                type="range" min="0.1" max="0.9" step="0.05"
                value={ratio} onChange={e => setRatio(e.target.value)}
                className="slider"
              />
              <div className="slider-ends">
                <span>Aggressive 10%</span>
                <span>Gentle 90%</span>
              </div>
            </div>
          ) : (
            <div className="budget-block">
              <label>Max output tokens</label>
              <input
                type="number" className="budget-in"
                placeholder="e.g. 200" value={budget}
                onChange={e => setBudget(e.target.value)} min="1"
              />
              {liveTokens > 0 && budget && (
                <span className="budget-pct">≈ {Math.round(parseInt(budget) / liveTokens * 100)}% of input</span>
              )}
            </div>
          )}

          <button className="go-btn" onClick={compress} disabled={loading || !text.trim()}>
            {loading
              ? <><Loader2 size={16} className="spin" /> Compressing…</>
              : <><Zap size={16} /> Compress Prompt <ChevronRight size={15} /></>
            }
          </button>

          {error && <div className="err-box">⚠ {error}</div>}

          {/* Algorithm explainer */}
          <div className="algo-box">
            <div className="algo-title">How it works</div>
            {[
              ["01", "Sentence segmentation + keyword extraction"],
              ["02", "TF-IDF scoring across the full corpus"],
              ["03", "Greedy selection by information density"],
              ["04", "Dynamic redundancy filter after each pick"],
              ["05", "Anchor first + last sentence for context"],
            ].map(([n, t]) => (
              <div key={n} className="algo-step">
                <span className="algo-n">{n}</span>
                <span>{t}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── Right: Results ───────────────────────────── */}
        <section className="pane pane-right">
          {!result ? (
            <div className="empty">
              <div className="empty-icon"><Zap size={48} strokeWidth={1.5} /></div>
              <p className="empty-title">Ready to compress</p>
              <p className="empty-sub">Enter a prompt and set your token budget or compression ratio, then hit <em>Compress</em>.</p>
              <div className="empty-targets">
                <span>Target: 30–60% token reduction</span>
                <span>·</span>
                <span>~70–85% meaning preserved</span>
              </div>
            </div>
          ) : (
            <>
              {/* Stats row */}
              <div className="stats-row">
                <StatCard label="Original" value={s.original_tokens.toLocaleString()} sub="tokens" color="#f87171" />
                <div className="arrow">→</div>
                <StatCard label="Compressed" value={s.compressed_tokens.toLocaleString()} sub="tokens" color="#4ade80" />
                <StatCard label="Saved" value={`${s.reduction_pct}%`} sub={`${s.tokens_saved} tokens`} color="#60a5fa" />
                <StatCard
                  label="Similarity"
                  value={result.similarity ? `${result.similarity.meaning_preserved_pct}%` : "—"}
                  sub="meaning kept"
                  color="#c084fc"
                />
              </div>

              <ReductionBar pct={s.reduction_pct} />

              {/* Tabs */}
              <div className="tabs">
                {[
                  { id: "output", icon: <Zap size={12} />, label: "Output" },
                  { id: "chart",  icon: <BarChart3 size={12} />, label: "Chart" },
                  { id: "sents",  icon: <List size={12} />, label: "Sentences" },
                  { id: "debug",  icon: <Bug size={12} />, label: "Debug" },
                ].map(t => (
                  <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
                    {t.icon}{t.label}
                  </button>
                ))}
              </div>

              <div className="tab-body">
                {tab === "output" && (
                  <div className="out-tab">
                    <div className="out-hdr">
                      <span>Compressed Prompt</span>
                      <CopyButton text={result.compressed_text} />
                    </div>
                    <div className="out-text">{result.compressed_text}</div>
                    <div className="out-meta">
                      <span>{s.compressed_sentence_count}/{s.original_sentence_count} sentences</span>
                      <span>·</span>
                      <span>{s.sentence_retention_pct}% retention</span>
                      {result.similarity && <>
                        <span>·</span>
                        <span>{result.similarity.method}</span>
                      </>}
                    </div>
                  </div>
                )}

                {tab === "chart" && (
                  <div className="chart-tab">
                    <TokenChart original={s.original_tokens} compressed={s.compressed_tokens} />
                    <div className="detail-grid">
                      {[
                        ["Token backend", s.backend],
                        ["Compression ratio", s.compression_ratio],
                        ["Budget used", `${s.budget_used} / ${s.budget_total}`],
                        ["Greedy iterations", s.iterations],
                        ["Sentences kept", `${s.compressed_sentence_count} / ${s.original_sentence_count}`],
                        ["Sentence retention", `${s.sentence_retention_pct}%`],
                      ].map(([k, v]) => (
                        <div key={k} className="detail-row">
                          <span>{k}</span><strong>{v}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {tab === "sents" && (
                  <SentenceBreakdown
                    allSentences={result.all_sentences}
                    selectedSentences={result.selected_sentences}
                  />
                )}

                {tab === "debug" && <DebugLog log={result.debug_log} />}
              </div>
            </>
          )}
        </section>
      </main>

      <footer className="foot">
        <span>TokenShrink v1.0</span>
        <span>·</span>
        <span>Python + Flask + React + Recharts</span>
        <span>·</span>
        <span>Greedy Search · TF-IDF · Dynamic Redundancy Filter</span>
      </footer>
    </div>
  );
}
