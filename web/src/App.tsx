import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ChatMessage, ChatReply, Metrics, PatientDetail, PatientRow, RiskBand } from "./api";

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

const CONDITIONS = [
  ["", "All conditions"],
  ["chf", "Heart failure"],
  ["copd", "COPD"],
  ["diabetes", "Diabetes"],
  ["ckd", "Kidney disease"],
] as const;

const SUGGESTIONS = [
  "Which heart failure patients should we call first, and why?",
  "How does readmission differ by discharge disposition?",
];

export default function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [band, setBand] = useState<RiskBand | "">("high");
  const [condition, setCondition] = useState("");
  const [rows, setRows] = useState<PatientRow[]>([]);
  const [listError, setListError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<PatientDetail | null>(null);
  const [detailError, setDetailError] = useState("");

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setMetrics(null));
  }, []);

  useEffect(() => {
    let stale = false;
    setListError("");
    api
      .patients(band, condition)
      .then((r) => !stale && setRows(r))
      .catch((e: Error) => !stale && setListError(e.message));
    return () => {
      stale = true;
    };
  }, [band, condition]);

  useEffect(() => {
    if (!selected) return;
    let stale = false;
    setDetail(null);
    setDetailError("");
    api
      .patient(selected)
      .then((d) => !stale && setDetail(d))
      .catch((e: Error) => !stale && setDetailError(e.message));
    return () => {
      stale = true;
    };
  }, [selected]);

  return (
    <div className="app">
      <header>
        <h1>CareSignal</h1>
        <span className="tag">{metrics ? metrics.data_label : "Loading data source…"} · decision support only</span>
        {metrics && (
          <span className="stats">
            {metrics.n_patients.toLocaleString()} discharges ·{" "}
            {metrics.observed_readmission_rate == null
              ? "readmissions not yet known"
              : `${pct(metrics.observed_readmission_rate)} readmitted in 30 days`}
            {metrics.test_auroc != null && ` · model AUROC ${metrics.test_auroc.toFixed(2)} on held-out data`}
          </span>
        )}
      </header>

      <main>
        <section className="panel list">
          <div className="filters">
            <select aria-label="Risk band" value={band} onChange={(e) => setBand(e.target.value as RiskBand | "")}>
              <option value="">All risk bands</option>
              <option value="high">High risk</option>
              <option value="medium">Medium risk</option>
              <option value="low">Low risk</option>
            </select>
            <select aria-label="Condition" value={condition} onChange={(e) => setCondition(e.target.value)}>
              {CONDITIONS.map(([v, label]) => (
                <option key={v} value={v}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          {listError ? (
            <p className="error">Couldn't load patients: {listError}</p>
          ) : rows.length === 0 ? (
            <p className="muted">No patients match these filters.</p>
          ) : (
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Age</th>
                    <th>Prior adm.</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.patient_id}
                      className={r.patient_id === selected ? "active" : ""}
                      tabIndex={0}
                      onClick={() => setSelected(r.patient_id)}
                      onKeyDown={(e) => e.key === "Enter" && setSelected(r.patient_id)}
                    >
                      <td>{r.patient_id}</td>
                      <td>
                        {r.age} {r.sex}
                      </td>
                      <td>{r.prior_admissions_12m}</td>
                      <td>
                        <span className={`band ${r.risk_band}`}>{pct(r.predicted_risk)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel detail">
          {!selected ? (
            <p className="muted">Select a patient to see their risk drivers and medication checks.</p>
          ) : detailError ? (
            <p className="error">Couldn't load {selected}: {detailError}</p>
          ) : !detail ? (
            <p className="muted">Loading {selected}…</p>
          ) : (
            <Detail d={detail} />
          )}
        </section>

        <Chat selected={selected} />
      </main>
    </div>
  );
}

function Detail({ d }: { d: PatientDetail }) {
  const { patient: p, risk } = d;
  const max = Math.max(...risk.top_drivers.map((x) => Math.abs(x.risk_contribution)), 0.001);
  const conditions = [
    p.chf && "Heart failure",
    p.copd && "COPD",
    p.diabetes && "Diabetes",
    p.ckd && "Kidney disease",
  ].filter(Boolean);

  return (
    <div className="scroll">
      <h2>
        {p.patient_id} <small>{p.age} y · {p.sex} · discharged to {p.discharge_disposition.replace("_", " ")}</small>
      </h2>
      <div className="score">
        <span className={`big band ${risk.risk_band}`}>{pct(risk.predicted_30d_readmission_risk)}</span>
        <span>
          30-day readmission risk ({risk.risk_band})<br />
          <small>Higher than {risk.cohort_percentile}% of the cohort</small>
        </span>
      </div>

      <h3>What's driving this estimate</h3>
      {risk.top_drivers.map((x) => (
        <div className="driver" key={x.feature}>
          <span>
            {x.label} <small>{x.value}</small>
          </span>
          <span className="bar">
            <i
              className={x.risk_contribution >= 0 ? "up" : "down"}
              style={{ width: `${(Math.abs(x.risk_contribution) / max) * 100}%` }}
            />
          </span>
          <span className="num">
            {x.risk_contribution >= 0 ? "+" : "−"}
            {(Math.abs(x.risk_contribution) * 100).toFixed(1)} pts
          </span>
        </div>
      ))}

      <h3>Clinical picture</h3>
      <dl>
        <dt>Conditions</dt>
        <dd>{conditions.length ? conditions.join(", ") : "None recorded"}</dd>
        <dt>Stay</dt>
        <dd>
          {p.length_of_stay} {p.length_of_stay === 1 ? "day" : "days"} · {p.prior_admissions_12m} admissions in prior 12 months
        </dd>
        <dt>Labs</dt>
        <dd>
          HbA1c {p.hba1c ?? "not recorded"} · creatinine {p.creatinine ?? "not recorded"} · SBP{" "}
          {p.systolic_bp ?? "not recorded"}
        </dd>
        <dt>Home</dt>
        <dd>{p.lives_alone == null ? "not recorded" : p.lives_alone ? "Lives alone" : "Lives with others"}</dd>
        <dt>Medications ({p.medications.length})</dt>
        <dd>{p.medications.join(", ")}</dd>
      </dl>

      <h3>Medication interactions</h3>
      {d.interactions.length === 0 ? (
        <p className="muted">No known interactions in this list.</p>
      ) : (
        <ul className="interactions">
          {d.interactions.map((i) => (
            <li key={i.drugs.join("+")}>
              <span className={`sev ${i.severity}`}>{i.severity}</span>
              <b>{i.drugs.join(" + ")}</b> — {i.effect}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

type Turn = ChatMessage & { tools?: ChatReply["tool_calls"] };

function Chat({ selected }: { selected: string | null }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and
    // React would try to call a returned value as the effect's cleanup.
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    const next: Turn[] = [...turns, { role: "user", content }];
    setTurns(next);
    setDraft("");
    setError("");
    setBusy(true);
    try {
      const r = await api.chat(next.map(({ role, content }) => ({ role, content })));
      setTurns([...next, { role: "assistant", content: r.reply, tools: r.tool_calls }]);
    } catch (e) {
      // Keep the user's message so they can retry without retyping.
      setTurns(turns);
      setDraft(content);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const suggestions = selected
    ? [`Summarise ${selected}'s readmission risk and suggest follow-up.`, ...SUGGESTIONS]
    : SUGGESTIONS;

  return (
    <section className="panel chat">
      <h2>Ask the agent</h2>
      <div className="scroll messages">
        {turns.length === 0 && (
          <div className="suggestions">
            {suggestions.map((s) => (
              <button key={s} onClick={() => send(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`msg ${t.role}`}>
            {t.tools && t.tools.length > 0 && (
              <div className="tools">Used: {[...new Set(t.tools.map((c) => c.name))].join(", ")}</div>
            )}
            {t.role === "assistant" ? <Markdown text={t.content} /> : t.content}
          </div>
        ))}
        {busy && <div className="msg assistant muted">Looking through the records…</div>}
        {error && <p className="error">The agent couldn't answer: {error}</p>}
        <div ref={end} />
      </div>
      <form
        onSubmit={(e: FormEvent) => {
          e.preventDefault();
          send(draft);
        }}
      >
        <input
          aria-label="Message"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask about a patient or the cohort"
          maxLength={8000}
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}

/** Minimal, dependency-free rendering of the markdown the model tends to
 *  produce: headings, bullets and bold/italic. Built from React nodes, so model
 *  output is never injected as HTML. */
function Markdown({ text }: { text: string }) {
  const inline = (line: string) =>
    line.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g).map((part, i) =>
      part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
        <b key={i}>{part.slice(2, -2)}</b>
      ) : part.startsWith("*") && part.endsWith("*") && part.length > 2 ? (
        <i key={i}>{part.slice(1, -1)}</i>
      ) : (
        part
      ),
    );
  return (
    <>
      {text.split("\n").map((line, i) => {
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        const bullet = line.match(/^\s*(?:[*-]|\d+\.)\s+(.*)$/);
        if (heading) return <div key={i} className="md-h">{inline(heading[1])}</div>;
        if (bullet) return <div key={i} className="md-li">{inline(bullet[1])}</div>;
        if (!line.trim()) return <div key={i} className="md-gap" />;
        return <div key={i}>{inline(line)}</div>;
      })}
    </>
  );
}
