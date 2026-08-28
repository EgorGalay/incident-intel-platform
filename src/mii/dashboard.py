from __future__ import annotations

import json
from html import escape

from .state import PhaseOneSnapshot


def render_dashboard(snapshot: PhaseOneSnapshot) -> str:
    payload = json.dumps(snapshot.to_dict(), ensure_ascii=True)
    incident = snapshot.incident
    incident_block = _render_incident(incident)
    anomalies_block = _render_anomalies(snapshot)
    metrics_block = _render_metrics(snapshot)
    drift_block = _render_drift(snapshot)
    quality_block = _render_quality(snapshot)
    graph_block = _render_graph(snapshot)
    rca_block = _render_rca(snapshot)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MII - Phase 3</title>
  <style>
    :root {{
      --bg: #09121f;
      --panel: #101c2e;
      --panel-2: #15233a;
      --text: #e7eef9;
      --muted: #92a5c0;
      --accent: #5dd6c6;
      --warning: #ffbf69;
      --danger: #ff6b6b;
      --border: rgba(255,255,255,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(93, 214, 198, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(255, 107, 107, 0.08), transparent 28%),
        linear-gradient(180deg, #07101b 0%, #0c1727 100%);
      color: var(--text);
    }}
    header {{
      padding: 32px 24px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(8, 15, 26, 0.72);
      backdrop-filter: blur(16px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .eyebrow {{
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
    }}
    h1 {{
      margin: 8px 0 8px;
      font-size: 34px;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
      padding: 20px 24px 36px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      grid-column: span 12;
      background: linear-gradient(180deg, rgba(21,35,58,0.96), rgba(16,28,46,0.96));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 48px rgba(0,0,0,0.2);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 18px;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 14px;
      background: rgba(255,255,255,0.03);
      border-radius: 14px;
      border: 1px solid var(--border);
    }}
    .metric .name {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .incident-critical {{
      color: var(--danger);
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .incident-high {{
      color: var(--warning);
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .two-col {{
      grid-column: span 12;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(93, 214, 198, 0.12);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    code {{
      color: #bfe9ff;
    }}
    @media (max-width: 900px) {{
      .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">ML Incident Intelligence Platform</div>
    <h1>Phase 3 Live Dashboard</h1>
    <p class="subtitle">Synthetic telemetry, anomaly detection, drift monitoring, dependency graphing, and RCA.</p>
  </header>
  <main>
    <section class="card">
      <h2>System Status</h2>
      <div class="status-grid">
        <div class="metric">
          <div class="name">Current Step</div>
          <div class="value" id="step">{snapshot.step}</div>
        </div>
        <div class="metric">
          <div class="name">Telemetry Signals</div>
          <div class="value" id="signal-count">{len(snapshot.latest_metrics)}</div>
        </div>
        <div class="metric">
          <div class="name">Recent Anomalies</div>
          <div class="value" id="anomaly-count">{len(snapshot.recent_anomalies)}</div>
        </div>
        <div class="metric">
          <div class="name">Incident State</div>
          <div class="value" id="incident-state">{'OPEN' if incident else 'WATCHING'}</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Active Incident</h2>
      <div id="incident-block">{incident_block}</div>
    </section>

    <section class="two-col">
      <div class="card">
        <h2>Latest Metrics</h2>
        {metrics_block}
      </div>
      <div class="card">
        <h2>Anomaly Feed</h2>
        {anomalies_block}
      </div>
    </section>

    <section class="two-col">
      <div class="card">
        <h2>Drift Findings</h2>
        {drift_block}
      </div>
      <div class="card">
        <h2>Data Quality</h2>
        {quality_block}
      </div>
    </section>

    <section class="two-col">
      <div class="card">
        <h2>Dependency Graph</h2>
        {graph_block}
      </div>
      <div class="card">
        <h2>Root Cause Analysis</h2>
        {rca_block}
      </div>
    </section>

    <section class="card">
      <h2>Snapshot</h2>
      <p class="muted">The dashboard refreshes automatically from <code>/api/state</code>.</p>
      <pre id="snapshot-json" style="white-space: pre-wrap; margin: 0;">{escape(payload)}</pre>
    </section>
  </main>

  <script>
    window.__MII_STATE__ = {payload};
    async function refreshState() {{
      try {{
        const response = await fetch('/api/state', {{ cache: 'no-store' }});
        if (!response.ok) return;
        const state = await response.json();
        document.getElementById('step').textContent = state.step;
        document.getElementById('signal-count').textContent = Object.keys(state.latest_metrics).length;
        document.getElementById('anomaly-count').textContent = state.recent_anomalies.length;
        document.getElementById('incident-state').textContent = state.incident ? 'OPEN' : 'WATCHING';
        document.getElementById('snapshot-json').textContent = JSON.stringify(state, null, 2);
      }} catch (error) {{
        console.error(error);
      }}
    }}
    setInterval(refreshState, 2000);
  </script>
</body>
</html>
"""


def _render_incident(incident) -> str:
    if incident is None:
        return '<p class="muted">No active incident yet. The system is still watching telemetry.</p>'

    anomaly_rows = "".join(
        f"<li><strong>{escape(anomaly.metric)}</strong> - {escape(anomaly.explanation)}</li>"
        for anomaly in incident.observed_anomalies
    )
    actions = "".join(f"<li>{escape(action)}</li>" for action in incident.recommended_actions)
    severity_class = f"incident-{incident.severity}" if incident.severity in {"critical", "high"} else "pill"

    return f"""
      <div class="{severity_class}">{escape(incident.severity.upper())}</div>
      <h3 style="margin-bottom: 8px;">{escape(incident.incident_id)} - {escape(incident.title)}</h3>
      <p class="muted">Detected at {escape(incident.detected_at.isoformat())}</p>
      <p><strong>Confidence:</strong> {incident.confidence:.2f}</p>
      <p>{escape(incident.summary)}</p>
      <div class="two-col" style="grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 0;">
        <div>
          <h4>Observed anomalies</h4>
          <ul>{anomaly_rows}</ul>
        </div>
        <div>
          <h4>Recommended actions</h4>
          <ul>{actions}</ul>
        </div>
      </div>
    """


def _render_anomalies(snapshot: PhaseOneSnapshot) -> str:
    if not snapshot.recent_anomalies:
        return '<p class="muted">No anomalies have been detected yet.</p>'

    rows = []
    for anomaly in snapshot.recent_anomalies[-8:][::-1]:
        rows.append(
            "<tr>"
            f"<td>{escape(anomaly.metric)}</td>"
            f"<td>{escape(anomaly.detector_name)}</td>"
            f"<td>{anomaly.z_score:.2f}</td>"
            f"<td>{escape(anomaly.severity)}</td>"
            f"<td>{escape(anomaly.explanation)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Metric</th><th>Detector</th><th>Score</th><th>Severity</th><th>Explanation</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _render_metrics(snapshot: PhaseOneSnapshot) -> str:
    if not snapshot.latest_metrics:
        return '<p class="muted">Waiting for the first telemetry frame.</p>'

    cards = []
    for metric, value in snapshot.latest_metrics.items():
        unit = snapshot.metric_units.get(metric, "")
        formatted = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
        suffix = f" {unit}" if unit else ""
        cards.append(
            f'<div class="metric"><div class="name">{escape(metric)}</div><div class="value">{formatted}{escape(suffix)}</div></div>'
        )
    return '<div class="status-grid">' + "".join(cards) + "</div>"


def _render_drift(snapshot: PhaseOneSnapshot) -> str:
    if not snapshot.drift_findings:
        return '<p class="muted">No drift alerts yet.</p>'

    rows = []
    for finding in snapshot.drift_findings[-6:][::-1]:
        rows.append(
            "<tr>"
            f"<td>{escape(finding.feature_name)}</td>"
            f"<td>{finding.psi:.3f}</td>"
            f"<td>{finding.current_mean:.2f}</td>"
            f"<td>{escape(finding.severity)}</td>"
            f"<td>{escape(finding.explanation)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Feature</th><th>PSI</th><th>Current mean</th><th>Severity</th><th>Explanation</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _render_quality(snapshot: PhaseOneSnapshot) -> str:
    if not snapshot.quality_issues:
        return '<p class="muted">No data-quality issues yet.</p>'

    rows = []
    for issue in snapshot.quality_issues[-6:][::-1]:
        rows.append(
            "<tr>"
            f"<td>{escape(issue.feature_name)}</td>"
            f"<td>{escape(issue.issue_type)}</td>"
            f"<td>{issue.rate:.1%}</td>"
            f"<td>{escape(issue.severity)}</td>"
            f"<td>{escape(issue.description)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Feature</th><th>Issue</th><th>Rate</th><th>Severity</th><th>Description</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _render_graph(snapshot: PhaseOneSnapshot) -> str:
    graph = snapshot.dependency_graph
    nodes = []
    for node in graph.nodes:
        children = ", ".join(graph.children(node.node_id)) or "none"
        nodes.append(
            "<tr>"
            f"<td>{escape(node.label)}</td>"
            f"<td>{escape(node.kind)}</td>"
            f"<td>{escape(node.team)}</td>"
            f"<td>{escape(children)}</td>"
            "</tr>"
        )

    edges = "".join(
        f"<li>{escape(edge.source)} → {escape(edge.target)} <span class='muted'>({escape(edge.relationship)})</span></li>"
        for edge in graph.edges
    )

    return (
        "<p class='muted'>Static dependency graph used for causal reasoning and blast-radius analysis.</p>"
        "<table><thead><tr><th>Component</th><th>Kind</th><th>Team</th><th>Downstream</th></tr></thead><tbody>"
        + "".join(nodes)
        + "</tbody></table><h4>Edges</h4><ul>"
        + edges
        + "</ul>"
    )


def _render_rca(snapshot: PhaseOneSnapshot) -> str:
    if not snapshot.root_cause_hypotheses:
        return '<p class="muted">No RCA hypotheses yet.</p>'

    blocks = []
    for hypothesis in snapshot.root_cause_hypotheses:
        reasons = "".join(f"<li>{escape(reason)}</li>" for reason in hypothesis.reasons)
        signals = ", ".join(hypothesis.evidence_signals) or "none"
        downstream = ", ".join(hypothesis.downstream_effects) or "none"
        blocks.append(
            f"""
            <div class="metric" style="margin-bottom: 12px;">
              <div class="name">{escape(hypothesis.label)}</div>
              <div class="value" style="font-size: 22px;">score {hypothesis.score:.2f}</div>
              <p><strong>Confidence:</strong> {hypothesis.confidence:.2f}</p>
              <p><strong>Signals:</strong> {escape(signals)}</p>
              <p><strong>Downstream effects:</strong> {escape(downstream)}</p>
              <ul>{reasons}</ul>
            </div>
            """
        )
    return "".join(blocks)
