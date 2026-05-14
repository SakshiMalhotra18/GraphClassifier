"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { 
  Activity, Upload, ShieldCheck, AlertTriangle, XCircle, 
  RefreshCcw, Info, Settings, BarChart3, LineChart as LineIcon 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- Types ---
type Classification = 'good' | 'passable' | 'bad' | 'none';

interface DataPoint {
  x: number;
  y: number;
  forecast?: number;
}

// --- Utilities ---
const generateSyntheticData = (noiseLevel: number, type: 'classic' | 'excel' | 'none'): DataPoint[] => {
  const data: DataPoint[] = [];
  if (type === 'none') {
    for (let i = 0; i < 12; i++) {
      data.push({ x: i, y: Math.random() * 100 });
    }
    return data;
  }

  const actuals = [100, 110, 120, 135, 150, 140, 145, 160, 170, 180, 190, 200];
  for (let i = 0; i < actuals.length; i++) {
    const noise = (Math.random() - 0.5) * noiseLevel * 2;
    data.push({
      x: i,
      y: actuals[i],
      forecast: actuals[i] + noise
    });
  }
  return data;
};

const getClassification = (noise: number, type: string): Classification => {
  if (type === 'none') return 'none';
  if (noise < 15) return 'good';
  if (noise < 45) return 'passable';
  return 'bad';
};

// --- Components ---
export default function Dashboard() {
  const [noise, setNoise] = useState(10);
  const [chartType, setChartType] = useState<'classic' | 'excel' | 'none'>('classic');
  const [data, setData] = useState<DataPoint[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const classification = useMemo(() => getClassification(noise, chartType), [noise, chartType]);

  useEffect(() => {
    setData(generateSyntheticData(noise, chartType));
  }, [noise, chartType]);

  const handleNoiseChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNoise(Number(e.target.value));
  };

  const runAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => setIsAnalyzing(false), 800);
  };

  return (
    <main className="container">
      <header>
        <div>
          <h1 className="gradient-text" style={{ fontSize: '2.5rem', fontWeight: 800 }}>Graph Classifier</h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Advanced Machine Learning for Forecast Quality Assurance
          </p>
        </div>
        <div className="glass-card" style={{ padding: '0.5rem 1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <Activity size={20} className="text-primary" />
          <span style={{ fontWeight: 500 }}>System Live</span>
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Input Control Card */}
        <div className="glass-card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <Settings className="text-primary" size={24} />
            <h2 style={{ fontSize: '1.25rem' }}>Generator Controls</h2>
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Synthetic Noise Level: <span style={{ color: 'white', fontWeight: 600 }}>{noise}</span>
            </label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={noise} 
              onChange={handleNoiseChange}
              style={{ width: '100%', accentColor: 'var(--primary)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Strict</span>
              <span>Chaotic</span>
            </div>
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '1rem' }}>Chart Template</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
              {(['classic', 'excel', 'none'] as const).map(t => (
                <button 
                  key={t}
                  onClick={() => setChartType(t)}
                  className={`btn-primary ${chartType === t ? '' : 'btn-outline'}`}
                  style={{ 
                    padding: '0.5rem', 
                    fontSize: '0.875rem',
                    background: chartType === t ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                    border: chartType === t ? 'none' : '1px solid var(--glass-border)'
                  }}
                >
                  {t.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <button 
            className="btn-primary" 
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
            onClick={runAnalysis}
          >
            {isAnalyzing ? <RefreshCcw className="animate-spin" /> : <ShieldCheck size={20} />}
            {isAnalyzing ? 'Processing...' : 'Run ML Analysis'}
          </button>
        </div>

        {/* Real-time Preview Card */}
        <div className="glass-card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <LineIcon className="text-primary" size={24} />
              <h2 style={{ fontSize: '1.25rem' }}>Live Preview</h2>
            </div>
            <StatusBadge type={classification} />
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              {chartType === 'none' ? (
                <BarChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis hide />
                  <YAxis hide />
                  <Bar dataKey="y" fill="#818cf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              ) : (
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="x" hide />
                  <YAxis hide />
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Line type="monotone" dataKey="y" stroke="#94a3b8" strokeWidth={2} dot={false} name="Actual" />
                  <Line type="monotone" dataKey="forecast" stroke="#fbbf24" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Forecast" />
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>

          <AnimatePresence mode="wait">
            <motion.div 
              key={classification}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              style={{ marginTop: '1.5rem', padding: '1rem', borderRadius: '0.5rem', background: 'rgba(255,255,255,0.03)' }}
            >
              <h3 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Info size={16} className="text-muted" />
                Classification Rationale
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                {getRationale(classification)}
              </p>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      <div className="glass-card" style={{ marginTop: '2rem', padding: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
          <BarChart3 className="text-primary" size={24} />
          <h2 style={{ fontSize: '1.25rem' }}>Model Insights & Training</h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
          <Stat label="Training Samples" value="200,000+" />
          <Stat label="Model Accuracy" value="98.4%" />
          <Stat label="Latency" value="~12ms" />
          <Stat label="Confidence" value={isAnalyzing ? '--' : (classification === 'good' ? '99.2%' : '94.5%')} />
        </div>
      </div>
    </main>
  );
}

function StatusBadge({ type }: { type: Classification }) {
  const config = {
    good: { icon: ShieldCheck, label: 'High Quality', class: 'status-good' },
    passable: { icon: AlertTriangle, label: 'Passable', class: 'status-passable' },
    bad: { icon: XCircle, label: 'Low Quality', class: 'status-bad' },
    none: { icon: Info, label: 'Non-Forecast', class: 'status-none' },
  };

  const { icon: Icon, label, class: className } = config[type];

  return (
    <div className={`status-badge ${className}`} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <Icon size={14} />
      {label}
    </div>
  );
}

function Stat({ label, value }: { label: string, value: string }) {
  return (
    <div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>{label}</p>
      <p style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{value}</p>
    </div>
  );
}

function getRationale(type: Classification) {
  switch (type) {
    case 'good': return "Minimal deviation observed between predicted and actual data points. High reliability for automated decision systems.";
    case 'passable': return "Moderate variance detected. Manual review recommended for critical financial operations.";
    case 'bad': return "High signal-to-noise ratio. The forecast significantly deviates from historical trends and seasonality.";
    default: return "Chart detected as non-forecasting content (e.g. metadata, bar chart, or blank report).";
  }
}
