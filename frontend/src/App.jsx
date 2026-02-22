import { useMemo, useState } from "react";

const RISK_COLORS = {
  Cyclone: "#0e7490",
  Temperature: "#2563eb",
  Rainfall: "#0891b2",
  Flood: "#0ea5e9",
  "Sea Level": "#38bdf8"
};

const DEFAULT_PROPERTY_ID = "98122";

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
    </div>
  );
}

function RiskPieChart({ annualPoints }) {
  const entries = Object.entries(annualPoints || {});
  const total = entries.reduce((sum, [, v]) => sum + Number(v || 0), 0);

  const gradient = useMemo(() => {
    if (!entries.length || total <= 0) return "conic-gradient(#d9e2ec 0deg 360deg)";
    let cursor = 0;
    const stops = entries.map(([riskName, value], idx) => {
      const sweep = (Number(value || 0) / total) * 360;
      const start = cursor;
      const end = cursor + sweep;
      cursor = end;
      const color = RISK_COLORS[riskName] || ["#0e7490", "#2563eb", "#0891b2", "#0ea5e9", "#38bdf8"][idx % 5];
      return `${color} ${start}deg ${end}deg`;
    });
    return `conic-gradient(${stops.join(", ")})`;
  }, [entries, total]);

  return (
    <div className="panel">
      <h3>Risk Composition (Pie Chart)</h3>
      <div className="pie-wrap">
        <div className="pie-chart" style={{ background: gradient }} />
        <div className="pie-legend">
          {entries.map(([k, v]) => (
            <div key={k} className="legend-row">
              <span className="legend-key">
                <span className="legend-dot" style={{ backgroundColor: RISK_COLORS[k] || "#5f6c7b" }} />
                {k}
              </span>
              <strong>{Number(v).toFixed(2)}</strong>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ProjectionChart({ series, startYear, tenureYears }) {
  if (!series?.length) return <p>No 50-year projection data available for this location.</p>;
  const width = 980;
  const height = 340;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 52;
  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;
  const minYear = Math.min(...series.map((p) => p.year));
  const maxYear = Math.max(...series.map((p) => p.year));
  const tenureEnd = startYear + Number(tenureYears) - 1;
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const xTicks = [minYear, minYear + 10, minYear + 20, minYear + 30, minYear + 40, maxYear]
    .filter((year, idx, arr) => year <= maxYear && arr.indexOf(year) === idx);

  const toX = (year) => padLeft + ((year - minYear) / Math.max(1, maxYear - minYear)) * chartWidth;
  const toY = (risk) => padTop + (1 - Number(risk)) * chartHeight;

  const fullPath = series.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.year)} ${toY(p.predicted_climate_risk)}`).join(" ");
  const tenureSeries = series.filter((p) => p.year <= tenureEnd);
  const tenurePath = tenureSeries
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.year)} ${toY(p.predicted_climate_risk)}`)
    .join(" ");

  return (
    <div className="panel">
      <h3>Projected Climate Risk (50-Year) with Tenure Highlight</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="projection-chart" role="img" aria-label="Projection chart">
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={padLeft}
              y1={toY(tick)}
              x2={width - padRight}
              y2={toY(tick)}
              stroke="#d2dee8"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
            <text x={padLeft - 10} y={toY(tick) + 4} textAnchor="end" className="chart-tick-label">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line x1={toX(tick)} y1={padTop} x2={toX(tick)} y2={height - padBottom} stroke="#e5edf3" strokeWidth="1" />
            <text x={toX(tick)} y={height - padBottom + 18} textAnchor="middle" className="chart-tick-label">
              {tick}
            </text>
          </g>
        ))}
        <line
          x1={padLeft}
          y1={height - padBottom}
          x2={width - padRight}
          y2={height - padBottom}
          stroke="#7f8c99"
          strokeWidth="1.2"
        />
        <line x1={padLeft} y1={padTop} x2={padLeft} y2={height - padBottom} stroke="#7f8c99" strokeWidth="1.2" />
        <path d={fullPath} fill="none" stroke="#145f7a" strokeWidth="2.2" />
        <path d={tenurePath} fill="none" stroke="#0f3057" strokeWidth="4" />
        <text x={width / 2} y={height - 10} textAnchor="middle" className="chart-axis-label">
          Year
        </text>
        <text x={14} y={height / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 14 ${height / 2})`}>
          Predicted Climate Risk (0 to 1)
        </text>
        <g transform={`translate(${width - 250}, ${padTop + 8})`}>
          <line x1="0" y1="0" x2="24" y2="0" stroke="#145f7a" strokeWidth="2.2" />
          <text x="30" y="4" className="chart-legend-label">
            50-Year Risk Projection
          </text>
          <line x1="0" y1="18" x2="24" y2="18" stroke="#0f3057" strokeWidth="4" />
          <text x="30" y="22" className="chart-legend-label">
            Tenure Risk Window
          </text>
        </g>
      </svg>
    </div>
  );
}

function toCsv(rows) {
  if (!rows?.length) return "";
  const headers = Object.keys(rows[0]);
  const escaped = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    if (s.includes(",") || s.includes('"') || s.includes("\n")) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  return [headers.join(","), ...rows.map((r) => headers.map((h) => escaped(r[h])).join(","))].join("\n");
}

export default function App() {
  const [propertyId, setPropertyId] = useState(DEFAULT_PROPERTY_ID);
  const [latitude, setLatitude] = useState(13.0827);
  const [longitude, setLongitude] = useState(80.2707);
  const [tenureYears, setTenureYears] = useState(15);
  const [loanAmount, setLoanAmount] = useState(5000000);
  const [projectionStartYear, setProjectionStartYear] = useState(2026);

  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [portfolioFile, setPortfolioFile] = useState(null);
  const [analyzingPortfolio, setAnalyzingPortfolio] = useState(false);
  const [portfolioError, setPortfolioError] = useState("");
  const [portfolioResult, setPortfolioResult] = useState(null);

  async function runPrediction() {
    setPredictionError("");
    setPrediction(null);
    setPredicting(true);
    try {
      const res = await fetch("/api/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: Number(latitude),
          longitude: Number(longitude),
          tenure_years: Number(tenureYears),
          loan_amount: Number(loanAmount),
          property_id: String(propertyId || DEFAULT_PROPERTY_ID),
          projection_start_year: Number(projectionStartYear)
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not run prediction.");
      setPrediction(data);
    } catch (err) {
      setPredictionError(String(err.message || err));
    } finally {
      setPredicting(false);
    }
  }

  async function runPortfolioAnalysis() {
    if (!portfolioFile) {
      setPortfolioError("Upload Portfolio CSV first.");
      return;
    }
    setPortfolioError("");
    setPortfolioResult(null);
    setAnalyzingPortfolio(true);
    try {
      const formData = new FormData();
      formData.append("file", portfolioFile);
      formData.append("projection_start_year", String(projectionStartYear));
      const res = await fetch("/api/portfolio/analyze/", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not process uploaded CSV.");
      setPortfolioResult(data);
    } catch (err) {
      setPortfolioError(String(err.message || err));
    } finally {
      setAnalyzingPortfolio(false);
    }
  }

  function downloadPortfolioCsv() {
    if (!portfolioResult?.results?.length) return;
    const blob = new Blob([toCsv(portfolioResult.results)], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "portfolio_analysis_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const score = prediction?.climate_credit_score ?? null;
  const propertyScoreOutput = prediction?.output_statements?.property_climate_credit_score || "Property-ID output not available.";
  const pricingOutput = prediction?.output_statements?.loan_pricing_adjustment || "Loan Pricing Adjustment: Not available.";
  const portfolioAlertOutput =
    portfolioResult?.portfolio_risk_alert ||
    prediction?.output_statements?.portfolio_risk_alert ||
    "Portfolio Risk Alert: Not available.";
  const explainabilityOutput = prediction?.output_statements?.explainability_log || "Explainability Log: Not available.";

  return (
    <div className="app">
      <header className="hero">
        <h1>Climate Loan Risk Assistant</h1>
        <p>High-clarity underwriting view with 50-year climate projection and policy-based risk controls.</p>
      </header>

      <div className="layout">
        <aside className="panel">
          <h2>Loan Inputs</h2>

          <label>
            Property ID
            <input type="text" value={propertyId} onChange={(e) => setPropertyId(e.target.value)} />
          </label>

          <label>
            Latitude
            <input type="number" value={latitude} step="0.000001" onChange={(e) => setLatitude(Number(e.target.value))} />
          </label>

          <label>
            Longitude
            <input type="number" value={longitude} step="0.000001" onChange={(e) => setLongitude(Number(e.target.value))} />
          </label>

          <label>
            Tenure Period (Years)
            <input type="number" min="1" max="50" value={tenureYears} onChange={(e) => setTenureYears(Number(e.target.value))} />
          </label>

          <label>
            Loan Amount
            <input type="number" min="0" value={loanAmount} onChange={(e) => setLoanAmount(Number(e.target.value))} />
          </label>

          <label>
            Projection Start Year
            <input
              type="number"
              min="2025"
              max="2100"
              value={projectionStartYear}
              onChange={(e) => setProjectionStartYear(Number(e.target.value))}
            />
          </label>

          <button className="primary" onClick={runPrediction} disabled={predicting}>
            {predicting ? "Running..." : "Run Prediction"}
          </button>
        </aside>

        <main className="content">
          {!prediction && !predictionError && <p className="info">Fill the inputs and click Run Prediction.</p>}
          {predictionError && <p className="error">{predictionError}</p>}

          {prediction && (
            <>
              <div className="grid-4">
                <MetricCard label="Climate Credit Score" value={score === null ? "N/A" : `${score}/100`} />
                <MetricCard label="Tenure Risk" value={`${prediction.tenure.tenure_risk_percent.toFixed(2)}%`} />
                <MetricCard label="Safety Status" value={prediction.safety_status} />
                <MetricCard label="Property ID" value={propertyId} />
              </div>

              <h3>Climate Credit Risk Analysis</h3>
              <ul className="output-list">
                <li>{propertyScoreOutput}</li>
                <li>{pricingOutput}</li>
                <li>{portfolioAlertOutput}</li>
                <li>{explainabilityOutput}</li>
              </ul>

              <div className="grid-3">
                <MetricCard label="Elevation (m)" value={Math.round(prediction.elevation_m)} />
                <MetricCard label="Latitude" value={prediction.latitude.toFixed(4)} />
                <MetricCard label="Longitude" value={prediction.longitude.toFixed(4)} />
              </div>

              <h3>Annual Risk Points</h3>
              <div className="grid-5">
                <MetricCard label="Cyclone" value={prediction.annual_points.Cyclone.toFixed(2)} />
                <MetricCard label="Temperature" value={prediction.annual_points.Temperature.toFixed(2)} />
                <MetricCard label="Rainfall" value={prediction.annual_points.Rainfall.toFixed(2)} />
                <MetricCard label="Flood" value={prediction.annual_points.Flood.toFixed(2)} />
                <MetricCard label="Sea Level" value={prediction.annual_points["Sea Level"].toFixed(2)} />
              </div>

              <RiskPieChart annualPoints={prediction.annual_points} />

              <h3>50-Year Projection and Tenure Risk Graph</h3>
              <ProjectionChart
                series={prediction.series_50}
                startYear={Number(projectionStartYear)}
                tenureYears={Number(tenureYears)}
              />

              <p className={prediction.safe ? "ok" : "error"}>
                {prediction.safe
                  ? "Tenure prediction indicates this loan is climate-safe under current thresholds."
                  : "Tenure prediction indicates this loan is not climate-safe under current thresholds."}
              </p>

              <details>
                <summary>Show 50-Year Projection Table</summary>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Year</th>
                        <th>Predicted Climate Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.series_50.map((r) => (
                        <tr key={r.year}>
                          <td>{r.year}</td>
                          <td>{r.predicted_climate_risk.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          )}
        </main>
      </div>

      <section className="panel">
        <h2>Portfolio Risk Analysis (CSV Upload)</h2>
        <p>
          Use this mode for large datasets. Required columns: property_id, latitude, longitude, tenure_years.
        </p>
        <div className="row">
          <input type="file" accept=".csv" onChange={(e) => setPortfolioFile(e.target.files?.[0] || null)} />
          <button className="primary" onClick={runPortfolioAnalysis} disabled={analyzingPortfolio}>
            {analyzingPortfolio ? "Processing..." : "Run Portfolio Analysis"}
          </button>
          <button onClick={downloadPortfolioCsv} disabled={!portfolioResult?.results?.length}>
            Download Results CSV
          </button>
        </div>
        {portfolioError && <p className="error">{portfolioError}</p>}

        {portfolioResult && (
          <>
            <div className="grid-4">
              <MetricCard label="Total Records" value={portfolioResult.total_records} />
              <MetricCard
                label="Average Tenure Risk"
                value={
                  portfolioResult.average_tenure_risk === null ? "N/A" : `${portfolioResult.average_tenure_risk.toFixed(2)}%`
                }
              />
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {Object.keys(portfolioResult.results[0] || {}).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {portfolioResult.results.map((row, idx) => (
                    <tr key={`${row["Property id"] || idx}-${idx}`}>
                      {Object.keys(portfolioResult.results[0] || {}).map((k) => (
                        <td key={`${idx}-${k}`}>{row[k] === null ? "" : String(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
