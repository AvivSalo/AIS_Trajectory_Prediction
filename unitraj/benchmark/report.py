"""
Evaluation Report Generator
Produces a standalone HTML dashboard summarising an evaluation run.
"""
import os
import json


def create_report(report_data: dict, output_dir: str) -> str:
    """
    Generate an HTML evaluation report with Chart.js visualizations.

    Parameters
    ----------
    report_data : dict
        Keys (all pre-computed by EvaluationCallback):
          timestamp, exp_name, ckpt_path, stride, past_len, future_len,
          eval_batch_size, load_num_workers, val_data_path,
          total_samples, n_vessels, n_batches,
          eval_duration_s, ms_per_sample, ms_per_batch, samples_per_sec,
          mean_ade, std_ade, p50_ade, p90_ade, p95_ade,
          mean_fde, std_fde, p50_fde, p90_fde, p95_fde,
          miss_2m, miss_4m, miss_8m,
          per_bin  : list[{label, count, ade, fde, miss, miss_4m, miss_8m}]
          per_vessel: list[{vessel_id, samples, ade, fde, miss, p90_fde, html_file}]
          cdf_x, cdf_y          : FDE CDF curve (150 pts)
          hist_labels, hist_counts : ADE histogram
    output_dir : str
        Directory where per-vessel HTMLs were written; report goes here too.

    Returns
    -------
    str  - absolute path to the generated report HTML.
    """
    rd = report_data

    # ── Pre-serialise chart data as JSON ────────────────────────────────────
    per_bin = rd.get('per_bin', [])
    per_vessel = rd.get('per_vessel', [])

    bin_labels_js    = json.dumps([b['label'] for b in per_bin])
    bin_ade_js       = json.dumps([round(b['ade'], 3) for b in per_bin])
    bin_fde_js       = json.dumps([round(b['fde'], 3) for b in per_bin])
    bin_miss2_js     = json.dumps([round(b['miss'], 1) for b in per_bin])
    bin_miss4_js     = json.dumps([round(b.get('miss_4m', 0), 1) for b in per_bin])
    bin_miss8_js     = json.dumps([round(b.get('miss_8m', 0), 1) for b in per_bin])

    sorted_vessels = sorted(per_vessel, key=lambda x: x['ade'])
    vessel_names_js = json.dumps([v['vessel_id'].replace('ais_', '') for v in sorted_vessels])
    vessel_ade_js   = json.dumps([round(v['ade'], 3) for v in sorted_vessels])
    vessel_fde_js   = json.dumps([round(v['fde'], 3) for v in sorted_vessels])

    cdf_x_js        = json.dumps(rd.get('cdf_x', []))
    cdf_y_js        = json.dumps(rd.get('cdf_y', []))

    hist_labels_js  = json.dumps(rd.get('hist_labels', []))
    hist_counts_js  = json.dumps(rd.get('hist_counts', []))

    # ── Per-bin table rows ───────────────────────────────────────────────────
    bin_rows_html = ""
    for b in per_bin:
        miss_color = '#ef4444' if b['miss'] > 70 else '#f59e0b' if b['miss'] > 50 else '#22c55e'
        bin_rows_html += f"""
        <tr>
          <td>{b['label']}</td>
          <td class="num">{b['count']:,}</td>
          <td class="num">{b['ade']:.2f}</td>
          <td class="num">{b['fde']:.2f}</td>
          <td class="num" style="color:{miss_color}">{b['miss']:.1f}%</td>
        </tr>"""

    # ── Per-vessel table rows ────────────────────────────────────────────────
    vessel_rows_html = ""
    for v in sorted_vessels:
        miss_color = '#ef4444' if v['miss'] > 70 else '#f59e0b' if v['miss'] > 50 else '#22c55e'
        link = f'<a href="{v["html_file"]}" target="_blank" class="viz-link">Open map ↗</a>' if v.get('html_file') else '—'
        short_id = v['vessel_id'].replace('ais_', '')
        vessel_rows_html += f"""
        <tr>
          <td class="vessel-name" title="{v['vessel_id']}">{short_id}</td>
          <td class="num">{v['samples']:,}</td>
          <td class="num">{v['ade']:.2f}</td>
          <td class="num">{v['fde']:.2f}</td>
          <td class="num">{v['p90_fde']:.2f}</td>
          <td class="num" style="color:{miss_color}">{v['miss']:.1f}%</td>
          <td>{link}</td>
        </tr>"""

    # ── Checkpoint short name ────────────────────────────────────────────────
    ckpt_short = os.path.basename(rd.get('ckpt_path', 'unknown'))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AIS_Wayformer_Report — {rd.get('exp_name', '')}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      font-family: 'Segoe UI', system-ui, Arial, sans-serif;
      background: #0f172a; color: #e2e8f0;
      font-size: 13px; line-height: 1.5;
    }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
      border-bottom: 1px solid #334155;
      padding: 20px 32px 16px;
    }}
    .header h1 {{
      margin: 0 0 4px; font-size: 20px; font-weight: 700; color: #f1f5f9;
    }}
    .header-meta {{
      display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px;
    }}
    .meta-item {{ display: flex; gap: 6px; align-items: baseline; }}
    .meta-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }}
    .meta-value {{ font-size: 12px; color: #94a3b8; font-weight: 500; }}
    .badge {{
      display: inline-block; background: #1d4ed8; color: #fff;
      font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;
    }}

    /* ── Layout ── */
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 32px 48px; }}

    /* ── Section headers ── */
    .section-title {{
      font-size: 10px; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: #64748b;
      margin: 28px 0 12px; padding-bottom: 6px;
      border-bottom: 1px solid #1e293b;
    }}

    /* ── Card grids ── */
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .card {{
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      padding: 14px 18px 12px; min-width: 130px; flex: 1;
    }}
    .card-value {{
      font-size: 26px; font-weight: 800; line-height: 1.1;
    }}
    .card-label {{
      font-size: 10px; color: #64748b; margin-top: 4px;
      text-transform: uppercase; letter-spacing: .06em;
    }}
    .card-sub {{
      font-size: 10px; color: #475569; margin-top: 2px;
    }}

    /* ── Info grid (config + timing) ── */
    .info-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 8px;
    }}
    .info-cell {{
      background: #1e293b; border: 1px solid #2d3f55; border-radius: 8px;
      padding: 10px 14px;
    }}
    .info-cell .lbl {{ font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: .07em; }}
    .info-cell .val {{ font-size: 13px; color: #cbd5e1; font-weight: 600; margin-top: 2px; word-break: break-all; }}

    /* ── Distribution row ── */
    .dist-table {{
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden;
    }}
    .dist-table th {{
      background: #0f172a; padding: 8px 14px;
      font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
      color: #64748b; font-weight: 700; text-align: right;
    }}
    .dist-table th:first-child {{ text-align: left; }}
    .dist-table td {{ padding: 9px 14px; border-top: 1px solid #1e293b; }}
    .dist-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .dist-table .metric-name {{ color: #94a3b8; font-weight: 600; }}

    /* ── Charts grid ── */
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .chart-box {{
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      padding: 16px 18px 14px;
    }}
    .chart-box.full-width {{
      grid-column: 1 / -1;
    }}
    .chart-title {{
      font-size: 11px; font-weight: 700; letter-spacing: .07em;
      text-transform: uppercase; color: #94a3b8; margin-bottom: 12px;
    }}
    .chart-canvas-wrap {{
      position: relative; height: 240px;
    }}
    .chart-canvas-wrap.tall {{
      height: 320px;
    }}

    /* ── Tables ── */
    .data-table {{
      width: 100%; border-collapse: separate; border-spacing: 0;
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      overflow: hidden;
    }}
    .data-table th {{
      background: #0f172a; padding: 8px 14px;
      font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
      color: #64748b; font-weight: 700; text-align: right;
    }}
    .data-table th:first-child {{ text-align: left; }}
    .data-table td {{ padding: 9px 14px; border-top: 1px solid #0f172a; color: #cbd5e1; }}
    .data-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .data-table td.vessel-name {{
      font-family: monospace; font-size: 12px; color: #94a3b8;
      max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .data-table tr:hover td {{ background: #263248; }}
    .viz-link {{
      color: #3b82f6; text-decoration: none; font-size: 11px; font-weight: 600;
    }}
    .viz-link:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>

<div class="header">
  <h1>📊 AIS_Wayformer_Report <span class="badge">Wayformer-AIS</span></h1>
  <div class="header-meta">
    <div class="meta-item"><span class="meta-label">Run</span><span class="meta-value">{rd.get('exp_name', '—')}</span></div>
    <div class="meta-item"><span class="meta-label">Checkpoint</span><span class="meta-value">{ckpt_short}</span></div>
    <div class="meta-item"><span class="meta-label">Generated</span><span class="meta-value">{rd.get('timestamp', '—')}</span></div>
    <div class="meta-item"><span class="meta-label">Data</span><span class="meta-value">{rd.get('val_data_path', '—')}</span></div>
  </div>
</div>

<div class="container">

  <!-- ── Overall Metrics ── -->
  <div class="section-title">Overall Metrics</div>
  <div class="cards">
    <div class="card">
      <div class="card-value" style="color:#3b82f6">{rd['mean_ade']:.2f} m</div>
      <div class="card-label">minADE6 (mean)</div>
      <div class="card-sub">σ={rd['std_ade']:.2f}  p90={rd['p90_ade']:.2f}</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#8b5cf6">{rd['mean_fde']:.2f} m</div>
      <div class="card-label">minFDE6 (mean)</div>
      <div class="card-sub">σ={rd['std_fde']:.2f}  p90={rd['p90_fde']:.2f}</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#ef4444">{rd['miss_2m']:.1f}%</div>
      <div class="card-label">Miss Rate @ 2 m</div>
      <div class="card-sub">FDE threshold = 2 m</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#f59e0b">{rd['miss_4m']:.1f}%</div>
      <div class="card-label">Miss Rate @ 4 m</div>
      <div class="card-sub">FDE threshold = 4 m</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#22c55e">{rd['miss_8m']:.1f}%</div>
      <div class="card-label">Miss Rate @ 8 m</div>
      <div class="card-sub">FDE threshold = 8 m</div>
    </div>
  </div>

  <!-- ── Distribution ── -->
  <div class="section-title">Error Distribution</div>
  <table class="dist-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Mean</th>
        <th>Std</th>
        <th>p50 (median)</th>
        <th>p90</th>
        <th>p95</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="metric-name" style="color:#3b82f6">minADE6</td>
        <td class="num">{rd['mean_ade']:.3f} m</td>
        <td class="num">{rd['std_ade']:.3f} m</td>
        <td class="num">{rd['p50_ade']:.3f} m</td>
        <td class="num">{rd['p90_ade']:.3f} m</td>
        <td class="num">{rd['p95_ade']:.3f} m</td>
      </tr>
      <tr>
        <td class="metric-name" style="color:#8b5cf6">minFDE6</td>
        <td class="num">{rd['mean_fde']:.3f} m</td>
        <td class="num">{rd['std_fde']:.3f} m</td>
        <td class="num">{rd['p50_fde']:.3f} m</td>
        <td class="num">{rd['p90_fde']:.3f} m</td>
        <td class="num">{rd['p95_fde']:.3f} m</td>
      </tr>
    </tbody>
  </table>

  <!-- ── Charts ── -->
  <div class="section-title">Charts</div>
  <div class="charts-grid">

    <!-- Chart 1: ADE/FDE by Time Offset -->
    <div class="chart-box">
      <div class="chart-title">ADE &amp; FDE by Prediction-Start Offset</div>
      <div class="chart-canvas-wrap">
        <canvas id="chartTimeBin"></canvas>
      </div>
    </div>

    <!-- Chart 2: Per-Vessel ADE Bar -->
    <div class="chart-box">
      <div class="chart-title">Per-Vessel Performance (minADE6 &amp; minFDE6)</div>
      <div class="chart-canvas-wrap">
        <canvas id="chartVessel"></canvas>
      </div>
    </div>

    <!-- Chart 3: FDE CDF -->
    <div class="chart-box">
      <div class="chart-title">FDE Cumulative Distribution (CDF)</div>
      <div class="chart-canvas-wrap">
        <canvas id="chartCDF"></canvas>
      </div>
    </div>

    <!-- Chart 4: ADE Histogram -->
    <div class="chart-box">
      <div class="chart-title">ADE Error Distribution (Histogram)</div>
      <div class="chart-canvas-wrap">
        <canvas id="chartHistADE"></canvas>
      </div>
    </div>

    <!-- Chart 5: Miss Rate at Multiple Thresholds by Time Bin -->
    <div class="chart-box full-width">
      <div class="chart-title">Miss Rate at Multiple Thresholds by Time Offset (@2m / @4m / @8m)</div>
      <div class="chart-canvas-wrap tall">
        <canvas id="chartMissRate"></canvas>
      </div>
    </div>

  </div><!-- /charts-grid -->

  <!-- ── Run Config ── -->
  <div class="section-title">Run Configuration</div>
  <div class="info-grid">
    <div class="info-cell"><div class="lbl">Stride</div><div class="val">{rd.get('stride', '—')} s</div></div>
    <div class="info-cell"><div class="lbl">History Length</div><div class="val">{rd.get('past_len', '—')} s</div></div>
    <div class="info-cell"><div class="lbl">Future Length (horizon)</div><div class="val">{rd.get('future_len', '—')} s</div></div>
    <div class="info-cell"><div class="lbl">Eval Batch Size</div><div class="val">{rd.get('eval_batch_size', '—')}</div></div>
    <div class="info-cell"><div class="lbl">Num Workers</div><div class="val">{rd.get('load_num_workers', '—')}</div></div>
    <div class="info-cell"><div class="lbl">Total Samples</div><div class="val">{rd.get('total_samples', 0):,}</div></div>
    <div class="info-cell"><div class="lbl">Vessels / Scenes</div><div class="val">{rd.get('n_vessels', '—')}</div></div>
    <div class="info-cell"><div class="lbl">Batches</div><div class="val">{rd.get('n_batches', 0):,}</div></div>
  </div>

  <!-- ── Inference Timing ── -->
  <div class="section-title">Inference Timing</div>
  <div class="cards">
    <div class="card">
      <div class="card-value" style="color:#06b6d4">{rd.get('eval_duration_s', 0):.1f} s</div>
      <div class="card-label">Total Inference Time</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#06b6d4">{rd.get('ms_per_sample', 0):.2f} ms</div>
      <div class="card-label">Per Sample</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#06b6d4">{rd.get('ms_per_batch', 0):.1f} ms</div>
      <div class="card-label">Per Batch</div>
    </div>
    <div class="card">
      <div class="card-value" style="color:#06b6d4">{rd.get('samples_per_sec', 0):.0f}</div>
      <div class="card-label">Samples / Second</div>
    </div>
  </div>

  <!-- ── Per Time Bin ── -->
  <div class="section-title">Performance by Prediction-Start Offset</div>
  <table class="data-table">
    <thead>
      <tr>
        <th>Time Offset</th>
        <th>Samples</th>
        <th>minADE6 (m)</th>
        <th>minFDE6 (m)</th>
        <th>Miss @ 2m</th>
      </tr>
    </thead>
    <tbody>{bin_rows_html}
    </tbody>
  </table>

  <!-- ── Per Vessel ── -->
  <div class="section-title">Per-Vessel / Per-Scene Breakdown</div>
  <table class="data-table">
    <thead>
      <tr>
        <th>Scene</th>
        <th>Samples</th>
        <th>minADE6 (m)</th>
        <th>minFDE6 (m)</th>
        <th>p90 FDE (m)</th>
        <th>Miss @ 2m</th>
        <th>Visualization</th>
      </tr>
    </thead>
    <tbody>{vessel_rows_html}
    </tbody>
  </table>

</div><!-- /container -->

<script>
(function() {{
  // ── Shared dark theme defaults ──────────────────────────────────────────
  const GRID_COLOR   = '#1e293b';
  const TICK_COLOR   = '#64748b';
  const TOOLTIP_BG   = '#0f172a';
  const TOOLTIP_TITLE = '#f1f5f9';
  const TOOLTIP_BODY  = '#cbd5e1';

  function darkOptions(extra) {{
    return Object.assign({{
      responsive: true,
      maintainAspectRatio: false,
      animation: {{ duration: 600 }},
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG,
          titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY,
          borderColor: '#334155',
          borderWidth: 1,
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: GRID_COLOR }},
          ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
        y: {{
          grid: {{ color: GRID_COLOR }},
          ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
      }},
    }}, extra || {{}});
  }}

  // ── Data ───────────────────────────────────────────────────────────────
  const binLabels  = {bin_labels_js};
  const binADE     = {bin_ade_js};
  const binFDE     = {bin_fde_js};
  const binMiss2   = {bin_miss2_js};
  const binMiss4   = {bin_miss4_js};
  const binMiss8   = {bin_miss8_js};

  const vesselNames = {vessel_names_js};
  const vesselADE   = {vessel_ade_js};
  const vesselFDE   = {vessel_fde_js};

  const cdfX = {cdf_x_js};
  const cdfY = {cdf_y_js};

  const histLabels = {hist_labels_js};
  const histCounts = {hist_counts_js};

  // ── Chart 1: ADE & FDE by Time Offset ─────────────────────────────────
  new Chart(document.getElementById('chartTimeBin'), {{
    type: 'line',
    data: {{
      labels: binLabels,
      datasets: [
        {{
          label: 'minADE6 (m)',
          data: binADE,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.12)',
          pointBackgroundColor: '#3b82f6',
          tension: 0.3,
          fill: false,
          borderWidth: 2,
          pointRadius: 4,
        }},
        {{
          label: 'minFDE6 (m)',
          data: binFDE,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139,92,246,0.12)',
          pointBackgroundColor: '#8b5cf6',
          tension: 0.3,
          fill: false,
          borderWidth: 2,
          pointRadius: 4,
        }},
      ],
    }},
    options: darkOptions({{
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG, titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY, borderColor: '#334155', borderWidth: 1,
          callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}} m` }},
        }},
      }},
      scales: {{
        x: {{ grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }}, maxRotation: 30 }} }},
        y: {{
          grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
          title: {{ display: true, text: 'Error (m)', color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
      }},
    }}),
  }});

  // ── Chart 2: Per-Vessel ADE & FDE (horizontal bar) ────────────────────
  new Chart(document.getElementById('chartVessel'), {{
    type: 'bar',
    data: {{
      labels: vesselNames,
      datasets: [
        {{
          label: 'minADE6 (m)',
          data: vesselADE,
          backgroundColor: 'rgba(59,130,246,0.75)',
          borderColor: '#3b82f6',
          borderWidth: 1,
          borderRadius: 3,
        }},
        {{
          label: 'minFDE6 (m)',
          data: vesselFDE,
          backgroundColor: 'rgba(139,92,246,0.75)',
          borderColor: '#8b5cf6',
          borderWidth: 1,
          borderRadius: 3,
        }},
      ],
    }},
    options: darkOptions({{
      indexAxis: 'y',
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG, titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY, borderColor: '#334155', borderWidth: 1,
          callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(3)}} m` }},
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
          title: {{ display: true, text: 'Error (m)', color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
        y: {{ grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 9 }} }} }},
      }},
    }}),
  }});

  // ── Chart 3: FDE CDF ───────────────────────────────────────────────────
  new Chart(document.getElementById('chartCDF'), {{
    type: 'line',
    data: {{
      labels: cdfX.map(v => v.toFixed(2)),
      datasets: [{{
        label: 'Cumulative % of samples',
        data: cdfY,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139,92,246,0.10)',
        fill: true,
        tension: 0.2,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
      }}],
    }},
    options: darkOptions({{
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG, titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY, borderColor: '#334155', borderWidth: 1,
          callbacks: {{
            title: items => `FDE = ${{items[0].label}} m`,
            label: ctx => ` ${{ctx.parsed.y.toFixed(1)}}% of samples`,
          }},
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: GRID_COLOR }},
          ticks: {{ color: TICK_COLOR, font: {{ size: 10 }}, maxTicksLimit: 10 }},
          title: {{ display: true, text: 'FDE (m)', color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
        y: {{
          grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
          title: {{ display: true, text: 'Cumulative %', color: TICK_COLOR, font: {{ size: 10 }} }},
          min: 0, max: 100,
        }},
      }},
    }}),
  }});

  // ── Chart 4: ADE Histogram ─────────────────────────────────────────────
  new Chart(document.getElementById('chartHistADE'), {{
    type: 'bar',
    data: {{
      labels: histLabels,
      datasets: [{{
        label: 'Samples',
        data: histCounts,
        backgroundColor: histCounts.map((_, i) => {{
          const t = i / (histLabels.length - 1);
          const r = Math.round(59 + t * (239 - 59));
          const g = Math.round(130 - t * (130 - 68));
          const b = Math.round(246 - t * (246 - 68));
          return `rgba(${{r}},${{g}},${{b}},0.8)`;
        }}),
        borderWidth: 0,
        borderRadius: 3,
      }}],
    }},
    options: darkOptions({{
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG, titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY, borderColor: '#334155', borderWidth: 1,
          callbacks: {{
            title: items => `ADE bucket: ${{items[0].label}} m`,
            label: ctx => ` ${{ctx.parsed.y.toLocaleString()}} samples`,
          }},
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: GRID_COLOR }},
          ticks: {{ color: TICK_COLOR, font: {{ size: 10 }}, maxRotation: 30 }},
          title: {{ display: true, text: 'ADE (m)', color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
        y: {{
          grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
          title: {{ display: true, text: 'Count', color: TICK_COLOR, font: {{ size: 10 }} }},
        }},
      }},
    }}),
  }});

  // ── Chart 5: Miss Rate at Multiple Thresholds by Time Bin ──────────────
  new Chart(document.getElementById('chartMissRate'), {{
    type: 'bar',
    data: {{
      labels: binLabels,
      datasets: [
        {{
          label: 'Miss @ 2m',
          data: binMiss2,
          backgroundColor: 'rgba(239,68,68,0.8)',
          borderColor: '#ef4444',
          borderWidth: 1,
          borderRadius: 3,
          stack: 'miss',
        }},
        {{
          label: 'Miss @ 4m (additional)',
          data: binMiss4.map((v, i) => Math.max(0, v - binMiss2[i])),
          backgroundColor: 'rgba(245,158,11,0.8)',
          borderColor: '#f59e0b',
          borderWidth: 1,
          borderRadius: 0,
          stack: 'miss',
        }},
        {{
          label: 'Miss @ 8m (additional)',
          data: binMiss8.map((v, i) => Math.max(0, v - binMiss4[i])),
          backgroundColor: 'rgba(34,197,94,0.8)',
          borderColor: '#22c55e',
          borderWidth: 1,
          borderRadius: 0,
          stack: 'miss',
        }},
      ],
    }},
    options: darkOptions({{
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG, titleColor: TOOLTIP_TITLE,
          bodyColor: TOOLTIP_BODY, borderColor: '#334155', borderWidth: 1,
          mode: 'index',
          callbacks: {{
            label: ctx => {{
              const label = ctx.dataset.label;
              const idx = ctx.dataIndex;
              if (label.startsWith('Miss @ 2m')) return ` Miss@2m: ${{binMiss2[idx].toFixed(1)}}%`;
              if (label.startsWith('Miss @ 4m')) return ` Miss@4m: ${{binMiss4[idx].toFixed(1)}}%`;
              if (label.startsWith('Miss @ 8m')) return ` Miss@8m: ${{binMiss8[idx].toFixed(1)}}%`;
              return '';
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          stacked: true,
          grid: {{ color: GRID_COLOR }},
          ticks: {{ color: TICK_COLOR, font: {{ size: 10 }}, maxRotation: 30 }},
        }},
        y: {{
          stacked: true,
          grid: {{ color: GRID_COLOR }}, ticks: {{ color: TICK_COLOR, font: {{ size: 10 }} }},
          title: {{ display: true, text: 'Miss Rate (%)', color: TICK_COLOR, font: {{ size: 10 }} }},
          min: 0,
        }},
      }},
    }}),
  }});

}})();
</script>

</body>
</html>"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path
