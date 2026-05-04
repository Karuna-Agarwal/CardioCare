import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import base64

# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open('svm_model.pkl', 'rb'))
model = load_model()

# ─── SHAP-style feature importance (model-agnostic perturbation method) ───────
def compute_feature_importance(model, input_df, n_repeats=1):
    """
    Model-agnostic permutation-based importance.
    For each feature, zero it out and measure prediction change.
    Works with ANY sklearn model — no SHAP dependency needed.
    """
    base_prob = _get_prob(model, input_df)
    importances = {}
    for col in input_df.columns:
        perturbed = input_df.copy()
        perturbed[col] = 0  # zero-out perturbation
        new_prob = _get_prob(model, perturbed)
        importances[col] = abs(base_prob - new_prob)
    total = sum(importances.values()) or 1
    return {k: v/total for k, v in importances.items()}

def _get_prob(model, df):
    try:    return model.predict_proba(df)[0][1]
    except: return float(model.predict(df)[0])

# ─── Risk stratification ──────────────────────────────────────────────────────
def stratify_risk(pct):
    if pct < 25:  return "Low",      "#10b981", "✅"
    if pct < 50:  return "Moderate", "#f59e0b", "⚡"
    if pct < 75:  return "High",     "#ef4444", "⚠️"
    return            "Critical",    "#dc2626", "🚨"

# ─── Population percentile (based on training dataset stats) ──────────────────
POP_STATS = {
    "Age":        {"mean": 53.5, "std": 9.4},
    "RestingBP":  {"mean": 132.4,"std": 18.5},
    "Cholesterol":{"mean": 198.8,"std": 109.4},
    "MaxHR":      {"mean": 136.8,"std": 25.5},
    "Oldpeak":    {"mean": 0.887,"std": 1.067},
}
def percentile_rank(feature, value):
    from scipy.special import ndtr
    s = POP_STATS.get(feature)
    if not s: return None
    z = (value - s["mean"]) / s["std"]
    return round(ndtr(z) * 100, 1)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CardioCare AI", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [('step',1),('history',[]),('predicted',False),('patient_name','')]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# MASTER CSS + ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --bg:       #05070f;
  --surface:  #0a0d18;
  --surface2: #0f1220;
  --surface3: #141828;
  --border:   #181d30;
  --border2:  #222840;
  --red:      #e63950;
  --red2:     #ff6b7a;
  --red-dim:  rgba(230,57,80,0.14);
  --red-glow: rgba(230,57,80,0.4);
  --green:    #10b981;
  --amber:    #f59e0b;
  --blue:     #60a5fa;
  --text:     #dde3f5;
  --muted:    #64708a;
  --faint:    #232840;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--faint); border-radius: 4px; }

/* ── Particle canvas ── */
#particle-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 0;
    opacity: 0.35;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    z-index: 10;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
    text-transform: none !important; font-size: 14px !important; letter-spacing: 0 !important;
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(160deg, #05070f 0%, #0a0d18 60%, #100510 100%);
    border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
    min-height: 240px; display: flex; align-items: center;
}
.hero-content { padding: 52px 68px; position: relative; z-index: 2; flex: 1; }
.hero-tag {
    font-size: 10px; font-weight: 600; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--red);
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}
.hero-tag::before { content:''; width:28px; height:1px; background:var(--red); }
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(28px,4vw,54px); font-weight: 800; line-height: 1.06;
    color: #eef2ff; margin-bottom: 14px;
    animation: hero-in 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.hero h1 em { color: var(--red); font-style: normal; position: relative; }
.hero h1 em::after {
    content: '';
    position: absolute; bottom: 2px; left: 0; right: 0;
    height: 2px; background: var(--red); opacity: 0.4;
    animation: underline-in 1s ease 0.6s both;
    transform-origin: left;
}
@keyframes underline-in { from{transform:scaleX(0)} to{transform:scaleX(1)} }
@keyframes hero-in { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
.hero-desc {
    font-size: 14px; color: var(--muted); max-width: 420px; line-height: 1.7;
    animation: hero-in 0.8s cubic-bezier(0.16,1,0.3,1) 0.15s both;
}
.hero-stats {
    display: flex; gap: 32px; margin-top: 28px; flex-wrap: wrap;
    animation: hero-in 0.8s cubic-bezier(0.16,1,0.3,1) 0.3s both;
}
.hstat { display:flex; flex-direction:column; gap:3px; }
.hstat-val { font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#eef2ff; }
.hstat-lbl { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; }

/* EKG */
.ekg-container {
    position: absolute; right: -20px; top: 0; bottom: 0;
    width: 55%; overflow: hidden; pointer-events: none;
}
.ekg-line {
    stroke-dasharray: 1400; stroke-dashoffset: 1400;
    animation: ekg-draw 2.8s cubic-bezier(0.4,0,0.2,1) forwards,
               ekg-beat 1.4s ease 2.8s infinite;
}
@keyframes ekg-draw { to { stroke-dashoffset: 0; } }
@keyframes ekg-beat { 0%,100%{opacity:0.18} 50%{opacity:0.06} }

/* Heartbeat orb */
.hb-orb {
    position: absolute; right: 10%; top: 50%; transform: translateY(-50%);
    width: 180px; height: 180px; border-radius: 50%; pointer-events: none;
    background: radial-gradient(circle, rgba(230,57,80,0.22) 0%, transparent 70%);
    animation: orb-pulse 1.8s ease infinite;
}
@keyframes orb-pulse { 0%,100%{transform:translateY(-50%) scale(1)} 50%{transform:translateY(-50%) scale(1.18)} }

/* ── Wizard ── */
.wizard-bar {
    display: flex; align-items: center;
    padding: 18px 68px;
    background: var(--surface); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 9;
    backdrop-filter: blur(12px);
}
.step-item { display:flex; align-items:center; gap:10px; flex:1; }
.step-item:not(:last-child)::after { content:''; flex:1; height:1px; background:var(--border2); margin:0 10px; }
.step-num {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'Syne',sans-serif; font-size:11px; font-weight:700;
    border:1.5px solid var(--border2); color:var(--muted);
    background:var(--surface2); flex-shrink:0; transition:all 0.3s;
}
.step-num.active {
    background: var(--red); border-color: var(--red); color: white;
    box-shadow: 0 0 0 4px var(--red-dim), 0 0 20px var(--red-glow);
    animation: step-pop 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
.step-num.done { background:var(--green); border-color:var(--green); color:white; }
@keyframes step-pop { from{transform:scale(0.6)} to{transform:scale(1)} }
.step-label { font-size:11px; font-weight:500; color:var(--muted); white-space:nowrap; }
.step-label.active { color:var(--text); font-weight:600; }

/* ── Page body ── */
.page-body { padding: 36px 68px; position: relative; z-index: 1; }

/* ── Section title ── */
.sec-title {
    font-family:'Syne',sans-serif; font-size:10px; font-weight:700;
    letter-spacing:0.18em; text-transform:uppercase; color:var(--muted);
    margin-bottom:16px; display:flex; align-items:center; gap:10px;
}
.sec-title::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Inputs ── */
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
.stNumberInput > div > div,
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-baseweb="input"] input, .stNumberInput input {
    color: var(--text) !important; font-family:'Instrument Sans',sans-serif !important;
    font-size:14px !important; background:transparent !important;
}
[data-baseweb="input"]:focus-within > div,
.stNumberInput:focus-within > div > div {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px var(--red-dim) !important;
}
label, .stSelectbox label, .stNumberInput label {
    color: var(--muted) !important; font-size:10px !important;
    font-weight:700 !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:13px !important; letter-spacing:0.05em !important;
    border-radius:10px !important; height:46px !important;
    border:none !important;
    background: linear-gradient(135deg, #e63950 0%, #c0172e 100%) !important;
    color: white !important;
    box-shadow: 0 4px 16px var(--red-dim) !important;
    transition: all 0.2s cubic-bezier(0.34,1.56,0.64,1) !important;
    position: relative; overflow: hidden !important;
}
.stButton > button::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
    opacity: 0; transition: opacity 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 28px var(--red-glow) !important;
}
.stButton > button:hover::after { opacity: 1 !important; }
.stButton > button:active { transform: translateY(0) scale(0.98) !important; }

/* ── Cards (staggered entrance) ── */
.card-enter {
    animation: card-in 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.card-enter:nth-child(1){animation-delay:0.05s}
.card-enter:nth-child(2){animation-delay:0.12s}
.card-enter:nth-child(3){animation-delay:0.19s}
.card-enter:nth-child(4){animation-delay:0.26s}
@keyframes card-in { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }

/* ── Metric grid ── */
.metric-grid {
    display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:28px;
}
.mc {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 18px;
    position: relative; overflow: hidden;
    animation: card-in 0.5s cubic-bezier(0.16,1,0.3,1) both;
    transition: border-color 0.2s, transform 0.2s;
}
.mc:hover { border-color: var(--border2); transform: translateY(-2px); }
.mc::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.mc.red::before   { background: linear-gradient(90deg,var(--red),var(--red2)); }
.mc.green::before { background: var(--green); }
.mc.amber::before { background: var(--amber); }
.mc.blue::before  { background: var(--blue); }
.mc-label { font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:6px; }
.mc-value { font-family:'Syne',sans-serif; font-size:26px; font-weight:800; line-height:1; margin-bottom:4px; }
.mc-sub   { font-size:11px; color:var(--muted); }
/* Animated counter */
.count-up { display:inline-block; }

/* ── Heartbeat ring (result page) ── */
.hb-ring-wrap {
    display: flex; justify-content: center; align-items: center;
    padding: 24px 0;
}
.hb-ring {
    width: 120px; height: 120px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    position: relative;
}
.hb-ring::before, .hb-ring::after {
    content: ''; position: absolute; inset: 0;
    border-radius: 50%; border: 2px solid currentColor;
    animation: ring-expand 2s ease infinite;
}
.hb-ring::after { animation-delay: 1s; }
@keyframes ring-expand {
    0%  { transform: scale(0.9); opacity: 0.8; }
    100%{ transform: scale(1.8); opacity: 0; }
}
.hb-ring-inner {
    font-size: 36px;
    animation: hb-beat 1.2s ease infinite;
}
@keyframes hb-beat {
    0%,100%{transform:scale(1)} 14%{transform:scale(1.3)} 28%{transform:scale(1)}
    42%{transform:scale(1.15)} 70%{transform:scale(1)}
}

/* ── Result banner ── */
.result-banner {
    border-radius: 16px; padding: 28px 34px; margin-bottom: 24px;
    animation: card-in 0.6s cubic-bezier(0.16,1,0.3,1) 0.1s both;
}
.result-banner.high { background:linear-gradient(135deg,rgba(230,57,80,0.1),rgba(176,32,53,0.05)); border:1px solid rgba(230,57,80,0.22); }
.result-banner.low  { background:linear-gradient(135deg,rgba(16,185,129,0.09),rgba(5,150,105,0.04)); border:1px solid rgba(16,185,129,0.18); }
.result-banner.mod  { background:linear-gradient(135deg,rgba(245,158,11,0.09),rgba(180,110,0,0.04)); border:1px solid rgba(245,158,11,0.18); }
.result-banner h2   { font-family:'Syne',sans-serif; font-size:24px; font-weight:800; margin-bottom:8px; }
.result-banner p    { font-size:13px; color:var(--muted); line-height:1.7; max-width:600px; }

/* ── What-if card ── */
.whatif-card {
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 14px; padding: 24px;
    animation: card-in 0.5s cubic-bezier(0.16,1,0.3,1) 0.3s both;
}
.whatif-title { font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:#eef2ff; margin-bottom:4px; }
.whatif-sub   { font-size:12px; color:var(--muted); margin-bottom:18px; }

/* ── Importance bars ── */
.imp-bar-wrap { margin-bottom: 10px; }
.imp-bar-label { display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; }
.imp-bar-label span:first-child { color:var(--text); font-weight:500; }
.imp-bar-label span:last-child  { color:var(--muted); font-family:'Syne',sans-serif; font-weight:600; }
.imp-bar-bg { height:6px; background:var(--border2); border-radius:100px; overflow:hidden; }
.imp-bar-fill { height:100%; border-radius:100px; transition:width 1.2s cubic-bezier(0.16,1,0.3,1); }

/* ── Population percentile ── */
.pct-wrap { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.pct-name { font-size:12px; color:var(--muted); width:140px; flex-shrink:0; }
.pct-bar-outer { flex:1; height:6px; background:var(--border2); border-radius:100px; overflow:hidden; }
.pct-bar-inner { height:100%; border-radius:100px; }
.pct-val  { font-size:11px; font-family:'Syne',sans-serif; font-weight:700; width:40px; text-align:right; flex-shrink:0; }

/* ── Tips ── */
.tip-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.tip-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
    transition: border-color 0.2s, transform 0.2s;
    animation: card-in 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.tip-card:hover { border-color:var(--border2); transform:translateY(-3px); }
.tip-icon  { font-size:24px; margin-bottom:8px; }
.tip-title { font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#eef2ff; margin-bottom:5px; }
.tip-body  { font-size:12px; color:var(--muted); line-height:1.55; }

/* ── History ── */
.hist-row {
    display:flex; align-items:center; gap:14px; padding:12px 16px;
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; margin-bottom:8px;
    transition: border-color 0.2s; cursor:default;
}
.hist-row:hover { border-color:var(--border2); }
.badge { display:inline-block; padding:2px 10px; border-radius:100px; font-size:10px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; }
.badge-r { background:var(--red-dim); color:var(--red); border:1px solid rgba(230,57,80,0.2); }
.badge-g { background:rgba(16,185,129,0.1); color:var(--green); border:1px solid rgba(16,185,129,0.2); }
.badge-a { background:rgba(245,158,11,0.1); color:var(--amber); border:1px solid rgba(245,158,11,0.2); }

/* ── Confetti canvas ── */
#confetti-canvas {
    position:fixed; top:0; left:0; width:100%; height:100%;
    pointer-events:none; z-index:9999;
}

/* ── Misc ── */
.div-line { height:1px; background:var(--border); margin:28px 0; }
.dl-btn {
    display:inline-flex; align-items:center; gap:8px;
    background:var(--surface2); border:1.5px solid var(--border2);
    color:var(--text); padding:10px 20px; border-radius:10px;
    font-size:13px; font-weight:600; cursor:pointer; text-decoration:none;
    transition:all 0.2s; font-family:'Instrument Sans',sans-serif;
}
.dl-btn:hover { border-color:var(--red); color:var(--red); text-decoration:none; }
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; }
.stAlert { border-radius:12px !important; }
.stProgress > div > div { background:var(--surface2) !important; border-radius:100px !important; }
.stProgress > div > div > div { background:linear-gradient(90deg,var(--green),var(--amber),var(--red)) !important; border-radius:100px !important; }
.footer { background:var(--surface); border-top:1px solid var(--border); padding:20px 68px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; font-size:11px; color:var(--faint); }
.footer strong { color:var(--muted); }

/* ── Tab overrides ── */
[data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; }
[data-baseweb="tab"]       { color:var(--muted) !important; font-family:'Instrument Sans',sans-serif !important; }
[aria-selected="true"][data-baseweb="tab"] { color:var(--text) !important; border-bottom:2px solid var(--red) !important; }

/* ── BMI display ── */
.bmi-box {
    margin-top:26px; background:var(--surface2); border:1.5px solid var(--border2);
    border-radius:10px; padding:12px 14px;
    animation: card-in 0.4s ease both;
}
.bmi-box .lbl { font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px; }
.bmi-box .val { font-family:'Syne',sans-serif;font-size:28px;font-weight:800;line-height:1; }
.bmi-box .cat { font-size:11px;margin-top:3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# JAVASCRIPT: Particles + Animated Counters + Confetti + Heartbeat speed
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<canvas id="particle-canvas"></canvas>
<canvas id="confetti-canvas"></canvas>

<script>
// ── Particle System ──────────────────────────────────────────────────────────
(function(){
  const canvas = document.getElementById('particle-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];
  
  function resize(){ W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor(){
      this.x = Math.random()*W; this.y = Math.random()*H;
      this.r = Math.random()*1.6+0.4;
      this.vx=(Math.random()-0.5)*0.25; this.vy=(Math.random()-0.5)*0.25;
      this.alpha = Math.random()*0.5+0.1;
      this.color = Math.random()>0.7 ? '#e63950' : '#3a4a7a';
    }
    update(){
      this.x+=this.vx; this.y+=this.vy;
      if(this.x<0||this.x>W) this.vx*=-1;
      if(this.y<0||this.y>H) this.vy*=-1;
    }
    draw(){
      ctx.beginPath(); ctx.arc(this.x,this.y,this.r,0,Math.PI*2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha; ctx.fill(); ctx.globalAlpha=1;
    }
  }
  
  for(let i=0;i<90;i++) particles.push(new Particle());

  // Draw connecting lines between nearby particles
  function drawLines(){
    for(let i=0;i<particles.length;i++){
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<110){
          ctx.beginPath();
          ctx.moveTo(particles[i].x,particles[i].y);
          ctx.lineTo(particles[j].x,particles[j].y);
          ctx.strokeStyle='rgba(100,112,138,'+((1-dist/110)*0.15)+')';
          ctx.lineWidth=0.5; ctx.stroke();
        }
      }
    }
  }

  function loop(){
    ctx.clearRect(0,0,W,H);
    drawLines();
    particles.forEach(p=>{ p.update(); p.draw(); });
    requestAnimationFrame(loop);
  }
  loop();
})();

// ── Animated Counter ──────────────────────────────────────────────────────────
window.animateCounter = function(el, target, duration=1400, suffix=''){
  if(!el) return;
  const start = performance.now();
  const from = 0;
  function step(now){
    const p = Math.min((now-start)/duration, 1);
    const ease = 1 - Math.pow(1-p, 4);
    el.textContent = Math.round(from + (target-from)*ease) + suffix;
    if(p<1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Confetti (for low risk) ────────────────────────────────────────────────
window.launchConfetti = function(){
  const canvas = document.getElementById('confetti-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width=window.innerWidth; canvas.height=window.innerHeight;
  const pieces = [];
  const colors=['#10b981','#34d399','#6ee7b7','#60a5fa','#a78bfa','#fbbf24','#f87171'];
  for(let i=0;i<160;i++){
    pieces.push({
      x:Math.random()*canvas.width, y:-20,
      r:Math.random()*6+3,
      color:colors[Math.floor(Math.random()*colors.length)],
      vx:(Math.random()-0.5)*5, vy:Math.random()*4+2,
      rot:Math.random()*360, vrot:(Math.random()-0.5)*8,
      alpha:1, shape:Math.random()>0.5?'rect':'circle'
    });
  }
  let frame=0;
  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    pieces.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.rot+=p.vrot;
      p.vy+=0.08; p.alpha-=0.006;
      if(p.alpha<=0) return;
      ctx.save(); ctx.globalAlpha=p.alpha;
      ctx.translate(p.x,p.y); ctx.rotate(p.rot*Math.PI/180);
      ctx.fillStyle=p.color;
      if(p.shape==='rect') ctx.fillRect(-p.r,-p.r/2,p.r*2,p.r);
      else { ctx.beginPath(); ctx.arc(0,0,p.r,0,Math.PI*2); ctx.fill(); }
      ctx.restore();
    });
    frame++;
    if(frame<220) requestAnimationFrame(draw);
    else ctx.clearRect(0,0,canvas.width,canvas.height);
  }
  draw();
}

// ── Animate importance bars on load ──────────────────────────────────────────
window.animateBars = function(){
  document.querySelectorAll('.imp-bar-fill').forEach(bar=>{
    const target = bar.getAttribute('data-width');
    bar.style.width = '0%';
    setTimeout(()=>{ bar.style.width = target; }, 100);
  });
}
setTimeout(window.animateBars, 400);
</script>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px;'>
        <div style='font-family:Syne,sans-serif;font-size:19px;font-weight:800;color:#eef2ff;'>🫀 CardioCare AI</div>
        <div style='font-size:10px;color:#64708a;margin-top:3px;letter-spacing:0.14em;text-transform:uppercase;'>Heart Risk Assessment v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio("", ["🩺  Assessment", "📊  Session History", "📚  Methodology"],
                   label_visibility="collapsed")

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    if st.session_state.history:
        total = len(st.session_state.history)
        high  = sum(1 for h in st.session_state.history if h['pred']==1)
        st.markdown(f"""
        <div style='font-size:10px;font-weight:700;color:#64708a;margin-bottom:8px;letter-spacing:0.1em;text-transform:uppercase;'>Session Summary</div>
        <div style='background:#0f1220;border:1px solid #181d30;border-radius:10px;padding:14px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px;'>
                <span style='color:#64708a;'>Total</span><strong style='color:#eef2ff;'>{total}</strong></div>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px;'>
                <span style='color:#64708a;'>High Risk</span><strong style='color:#e63950;'>{high}</strong></div>
            <div style='display:flex;justify-content:space-between;font-size:12px;'>
                <span style='color:#64708a;'>Low/Moderate</span><strong style='color:#10b981;'>{total-high}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear History", key="clr"):
            st.session_state.history=[]; st.rerun()

    st.markdown("""
    <div style='margin-top:28px;font-size:10px;color:#1e2438;line-height:1.8;border-top:1px solid #181d30;padding-top:16px;'>
    ⚠️ Educational use only.<br>Not a clinical diagnostic tool.<br>Always consult a physician.
    </div>
    """, unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-content">
    <div class="hero-tag">AI-Powered Cardiovascular Analysis</div>
    <h1>Predict your<br><em>heart disease risk</em><br>intelligently.</h1>
    <p class="hero-desc">Multi-step clinical assessment with real-time explainability, what-if simulation, and population benchmarking.</p>
    <div class="hero-stats">
      <div class="hstat"><span class="hstat-val">918</span><span class="hstat-lbl">Training Records</span></div>
      <div class="hstat"><span class="hstat-val">~87%</span><span class="hstat-lbl">Model Accuracy</span></div>
      <div class="hstat"><span class="hstat-val">~93%</span><span class="hstat-lbl">ROC-AUC</span></div>
      <div class="hstat"><span class="hstat-val">SVM</span><span class="hstat-lbl">Algorithm</span></div>
    </div>
  </div>
  <div class="ekg-container">
    <svg viewBox="0 0 900 240" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
      <path class="ekg-line"
        d="M0,120 L60,120 L80,120 L105,38 L125,202 L148,14 L168,206 L188,120
           L230,120 L300,120 L320,120 L345,38 L365,202 L388,14 L408,206 L428,120
           L470,120 L540,120 L560,120 L585,38 L605,202 L628,14 L648,206 L668,120
           L710,120 L780,120 L800,120 L825,38 L845,202 L868,14 L888,206 L900,120"
        fill="none" stroke="#e63950" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div class="hb-orb"></div>
</div>
""", unsafe_allow_html=True)

# ─── helpers ──────────────────────────────────────────────────────────────────
def norm(v, lo, hi): return min(max((v-lo)/(hi-lo), 0), 1)

# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT PAGE
# ══════════════════════════════════════════════════════════════════════════════
if "Assessment" in nav:
    step = st.session_state.step

    def sn(n):
        if n < step: return "done"
        if n == step: return "active"
        return ""

    st.markdown(f"""
    <div class="wizard-bar">
      <div class="step-item">
        <div class="step-num {sn(1)}">{'✓' if step>1 else '1'}</div>
        <div class="step-label {'active' if step==1 else ''}">Personal Info</div>
      </div>
      <div class="step-item">
        <div class="step-num {sn(2)}">{'✓' if step>2 else '2'}</div>
        <div class="step-label {'active' if step==2 else ''}">Clinical Vitals</div>
      </div>
      <div class="step-item">
        <div class="step-num {sn(3)}">{'✓' if step>3 else '3'}</div>
        <div class="step-label {'active' if step==3 else ''}">Cardiac Markers</div>
      </div>
      <div class="step-item" style="flex:0;min-width:fit-content;">
        <div class="step-num {sn(4)}">4</div>
        <div class="step-label {'active' if step==4 else ''}">Results & Analysis</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    # ─── STEP 1 ───────────────────────────────────────────────────────────────
    if step == 1:
        st.markdown('<div class="sec-title">Step 1 of 3 — Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_name = st.text_input("Patient Name / ID",
                value=st.session_state.patient_name, placeholder="e.g. Arjun S. or P-001")
        with c2:
            Age = st.number_input("Age (years)", 20, 100, 52)
        with c3:
            gender = st.selectbox("Biological Sex", ['Male','Female'])

        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">BMI Calculator</div>', unsafe_allow_html=True)
        b1,b2,b3 = st.columns(3)
        with b1: height_cm = st.number_input("Height (cm)", 100, 250, 170)
        with b2: weight_kg = st.number_input("Weight (kg)", 30, 200, 70)
        with b3:
            bmi = weight_kg / ((height_cm/100)**2)
            bmi_cat   = ("Underweight" if bmi<18.5 else "Normal" if bmi<25
                         else "Overweight" if bmi<30 else "Obese")
            bmi_color = ("#10b981" if bmi_cat=="Normal"
                         else "#f59e0b" if bmi_cat in ("Underweight","Overweight") else "#e63950")
            st.markdown(f"""<div class="bmi-box">
                <div class="lbl">Calculated BMI</div>
                <div class="val" style="color:{bmi_color};">{bmi:.1f}</div>
                <div class="cat" style="color:{bmi_color};">{bmi_cat}</div>
            </div>""", unsafe_allow_html=True)

        st.session_state['s1'] = dict(patient_name=patient_name, Age=Age,
                                       gender=gender, bmi=bmi, bmi_cat=bmi_cat)
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        _, nc = st.columns([3,1])
        with nc:
            if st.button("Next: Clinical Vitals →", key="n1"):
                st.session_state.patient_name = patient_name
                st.session_state.step = 2; st.rerun()

    # ─── STEP 2 ───────────────────────────────────────────────────────────────
    elif step == 2:
        s1 = st.session_state.get('s1', {})
        st.markdown(f"""<div style='font-size:12px;color:var(--muted);margin-bottom:18px;'>
            <strong style='color:var(--text);'>{s1.get('patient_name') or 'Anonymous'}</strong> ·
            Age {s1.get('Age','—')} · {s1.get('gender','—')} · BMI {s1.get('bmi',0):.1f} ({s1.get('bmi_cat','')})
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Step 2 of 3 — Clinical Vitals</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            RestingBP = st.number_input("Resting Blood Pressure (mmHg)", 0, 250, 125,
                help="Normal < 120. Elevated: 120–129. High ≥ 130.")
            pct_bp = percentile_rank("RestingBP", RestingBP)
            st.caption(f"{'🟢 Normal' if RestingBP<120 else '🟡 Elevated' if RestingBP<130 else '🔴 High'}  ·  Higher than {pct_bp}% of population")

            Cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 0, 650, 212,
                help="Desirable < 200. Borderline: 200–239. High ≥ 240.")
            pct_chol = percentile_rank("Cholesterol", Cholesterol)
            st.caption(f"{'🟢 Desirable' if Cholesterol<200 else '🟡 Borderline' if Cholesterol<240 else '🔴 High'}  ·  Higher than {pct_chol}% of population")

        with c2:
            FastingBS = st.selectbox("Fasting Blood Sugar", [0,1],
                format_func=lambda x: "🟢 Normal  (≤ 120 mg/dL)" if x==0 else "🔴 High  (> 120 mg/dL)")

            MaxHR = st.number_input("Max Heart Rate Achieved (bpm)", 60, 250, 168,
                help="Maximum HR during exercise stress test.")
            age_v = s1.get('Age', 52)
            hr_pct = int((MaxHR/max(220-age_v,1))*100)
            pct_hr = percentile_rank("MaxHR", MaxHR)
            st.caption(f"≈ {hr_pct}% of age-predicted max  ·  Higher than {pct_hr}% of population")

        st.session_state['s2'] = dict(RestingBP=RestingBP, Cholesterol=Cholesterol,
                                       FastingBS=FastingBS, MaxHR=MaxHR)
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        b1,b2,_ = st.columns([1,1,2])
        with b1:
            if st.button("← Back", key="b2"): st.session_state.step=1; st.rerun()
        with b2:
            if st.button("Next: Cardiac Markers →", key="n2"): st.session_state.step=3; st.rerun()

    # ─── STEP 3 ───────────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown('<div class="sec-title">Step 3 of 3 — Cardiac Markers & ECG</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            ChestPainType = st.selectbox("Chest Pain Type",
                ['ATA','NAP','ASY','TA'],
                format_func=lambda x: {
                    'ATA':'Atypical Angina (ATA)','NAP':'Non-Anginal Pain (NAP)',
                    'ASY':'Asymptomatic (ASY) ← Highest Risk','TA':'Typical Angina (TA)'
                }[x])
            RestingECG = st.selectbox("Resting ECG Result",
                ['Normal','ST','LVH'],
                format_func=lambda x:{
                    'Normal':'Normal','ST':'ST-T Wave Abnormality','LVH':'Left Ventricular Hypertrophy'
                }[x])
        with c2:
            ExerciseAngina = st.selectbox("Exercise-Induced Angina", ['N','Y'],
                format_func=lambda x: "✅ No" if x=='N' else "⚠️ Yes — pain during exertion")

            Oldpeak = st.number_input("ST Depression (Oldpeak)", -3.0, 10.0, 1.0, step=0.1,
                help="ST depression during exercise. > 2.0 is clinically significant.")
            pct_op = percentile_rank("Oldpeak", Oldpeak)
            st.caption(f"{'🟢 Low' if Oldpeak<=1 else '🟡 Moderate' if Oldpeak<=2 else '🔴 Significant'}  ·  Higher than {pct_op}% of population")

            ST_Slope = st.selectbox("ST Slope (Exercise ECG)", ['Up','Flat','Down'],
                format_func=lambda x:{
                    'Up':'⬆️ Upsloping — Favourable','Flat':'➡️ Flat — Borderline',
                    'Down':'⬇️ Downsloping — High Risk'
                }[x])

        st.session_state['s3'] = dict(ChestPainType=ChestPainType, RestingECG=RestingECG,
                                       ExerciseAngina=ExerciseAngina, Oldpeak=Oldpeak, ST_Slope=ST_Slope)
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        b1,b2,_ = st.columns([1,1,2])
        with b1:
            if st.button("← Back", key="b3"): st.session_state.step=2; st.rerun()
        with b2:
            if st.button("🔍 Run Full Assessment", key="n3"): st.session_state.step=4; st.rerun()

    # ─── STEP 4: RESULTS ──────────────────────────────────────────────────────
    elif step == 4:
        s1 = st.session_state.get('s1',{})
        s2 = st.session_state.get('s2',{})
        s3 = st.session_state.get('s3',{})

        Age            = s1.get('Age',52)
        gender         = s1.get('gender','Male')
        RestingBP      = s2.get('RestingBP',125)
        Cholesterol    = s2.get('Cholesterol',212)
        FastingBS      = s2.get('FastingBS',0)
        MaxHR          = s2.get('MaxHR',168)
        ChestPainType  = s3.get('ChestPainType','ATA')
        RestingECG     = s3.get('RestingECG','Normal')
        ExerciseAngina = s3.get('ExerciseAngina','N')
        Oldpeak        = s3.get('Oldpeak',1.0)
        ST_Slope       = s3.get('ST_Slope','Up')

        input_df = pd.DataFrame({
            'Age':[Age],'RestingBP':[RestingBP],'Cholesterol':[Cholesterol],
            'FastingBS':[FastingBS],'MaxHR':[MaxHR],'Oldpeak':[Oldpeak],
            'sex':[1 if gender=='Male' else 0],
            'exerciseAngina':[1 if ExerciseAngina=='Y' else 0],
            'RestingECG_LVH':[1 if RestingECG=='LVH' else 0],
            'RestingECG_Normal':[1 if RestingECG=='Normal' else 0],
            'RestingECG_ST':[1 if RestingECG=='ST' else 0],
            'ChestPainType_ASY':[1 if ChestPainType=='ASY' else 0],
            'ChestPainType_ATA':[1 if ChestPainType=='ATA' else 0],
            'ChestPainType_NAP':[1 if ChestPainType=='NAP' else 0],
            'ChestPainType_TA':[1 if ChestPainType=='TA' else 0],
            'st_Slope':[{'Up':0,'Down':1,'Flat':2}[ST_Slope]]
        })

        prediction = model.predict(input_df)[0]
        try:    prob = model.predict_proba(input_df)[0][1]
        except: prob = 0.72 if prediction==1 else 0.28
        pct = round(prob*100, 1)

        risk_label, risk_color, risk_icon = stratify_risk(pct)

        # Health age
        risk_add = sum([3*(RestingBP>130), 4*(Cholesterol>240), 3*(FastingBS==1),
                        4*(ExerciseAngina=='Y'), 3*(Oldpeak>2), 3*(ChestPainType=='ASY'),
                        2*(ST_Slope=='Down'), 2*(RestingECG=='LVH')])
        health_age = Age + (risk_add if prediction==1 else -2)
        delta_age  = health_age - Age

        # Feature importance
        imp = compute_feature_importance(model, input_df)
        # Map to readable names
        name_map = {
            'Age':'Age','RestingBP':'Resting BP','Cholesterol':'Cholesterol',
            'MaxHR':'Max Heart Rate','Oldpeak':'ST Depression',
            'sex':'Gender','exerciseAngina':'Exercise Angina',
            'RestingECG_LVH':'ECG (LVH)','RestingECG_Normal':'ECG (Normal)',
            'RestingECG_ST':'ECG (ST)','FastingBS':'Fasting Blood Sugar',
            'ChestPainType_ASY':'Chest Pain (ASY)','ChestPainType_ATA':'Chest Pain (ATA)',
            'ChestPainType_NAP':'Chest Pain (NAP)','ChestPainType_TA':'Chest Pain (TA)',
            'st_Slope':'ST Slope'
        }
        # Merge ECG and ChestPain sub-features
        merged_imp = {
            'Age':          imp.get('Age',0),
            'Resting BP':   imp.get('RestingBP',0),
            'Cholesterol':  imp.get('Cholesterol',0),
            'Max Heart Rate':imp.get('MaxHR',0),
            'ST Depression':imp.get('Oldpeak',0),
            'Blood Sugar':  imp.get('FastingBS',0),
            'Chest Pain Type': max(imp.get('ChestPainType_ASY',0), imp.get('ChestPainType_ATA',0),
                                   imp.get('ChestPainType_NAP',0), imp.get('ChestPainType_TA',0)),
            'ECG Result':   max(imp.get('RestingECG_LVH',0), imp.get('RestingECG_Normal',0), imp.get('RestingECG_ST',0)),
            'Exercise Angina': imp.get('exerciseAngina',0),
            'ST Slope':     imp.get('st_Slope',0),
        }
        sorted_imp = sorted(merged_imp.items(), key=lambda x: x[1], reverse=True)[:7]

        # Save history
        entry = dict(name=s1.get('patient_name') or 'Anonymous', age=Age, gender=gender,
                     pred=int(prediction), prob=pct, health_age=health_age,
                     risk_label=risk_label, time=datetime.now().strftime("%H:%M"),
                     bmi=s1.get('bmi',0), bmi_cat=s1.get('bmi_cat',''))
        if not st.session_state.predicted:
            st.session_state.history.append(entry)
            st.session_state.predicted = True

        banner_cls = "high" if pct>=60 else "mod" if pct>=35 else "low"
        name_d = entry['name']

        # ── Animated JS for this result ──
        confetti_js = "launchConfetti();" if pct < 35 else ""
        st.markdown(f"""
        <script>
        setTimeout(function(){{
          // Animate all .count-up elements
          document.querySelectorAll('.count-up').forEach(function(el){{
            var target = parseFloat(el.getAttribute('data-target'));
            var suffix = el.getAttribute('data-suffix')||'';
            animateCounter(el, target, 1500, suffix);
          }});
          // Animate importance bars
          animateBars();
          // Confetti if low risk
          {confetti_js}
          // Dynamic heartbeat speed based on risk
          var hbEl = document.querySelector('.hb-ring-inner');
          if(hbEl){{
            var dur = {max(0.5, 1.8 - prob*1.4):.2f};
            hbEl.style.animationDuration = dur+'s';
          }}
        }}, 300);
        </script>
        """, unsafe_allow_html=True)

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metric-grid">
          <div class="mc {'red' if pct>=50 else 'green'}">
            <div class="mc-label">Risk Level</div>
            <div class="mc-value" style="color:{risk_color};font-size:18px;">{risk_icon} {risk_label.upper()}</div>
            <div class="mc-sub">{pct}% probability</div>
          </div>
          <div class="mc {'red' if pct>=50 else 'amber' if pct>=35 else 'green'}">
            <div class="mc-label">Risk Score</div>
            <div class="mc-value" style="color:{risk_color};">
              <span class="count-up" data-target="{pct}" data-suffix="%">0%</span>
            </div>
            <div class="mc-sub">Heart disease likelihood</div>
          </div>
          <div class="mc {'red' if delta_age>0 else 'green'}">
            <div class="mc-label">Cardiac Health Age</div>
            <div class="mc-value" style="color:{'#e63950' if delta_age>0 else '#10b981'};">
              <span class="count-up" data-target="{health_age}" data-suffix=" yrs">{health_age} yrs</span>
            </div>
            <div class="mc-sub">{'▲ ' + str(abs(delta_age)) + ' yrs older than body' if delta_age>0 else '▼ Heart younger than age'}</div>
          </div>
          <div class="mc blue">
            <div class="mc-label">BMI</div>
            <div class="mc-value" style="color:#60a5fa;">{s1.get('bmi',0):.1f}</div>
            <div class="mc-sub">{s1.get('bmi_cat','—')}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Result banner + heartbeat ──
        r1, r2 = st.columns([5,1])
        with r1:
            st.markdown(f"""
            <div class="result-banner {banner_cls}">
              <h2 style="color:{risk_color};">{risk_icon} {risk_label} Risk of Heart Disease Detected</h2>
              <p>The SVM model predicts a <strong style="color:{risk_color};">{pct}%</strong> probability
              for <strong style="color:#eef2ff;">{name_d}</strong>.
              {'Significant clinical risk factors were identified. Immediate cardiology consultation is strongly recommended.'
               if pct>=60 else
               'Several moderate risk factors present. Lifestyle modifications and medical review are advised.'
               if pct>=35 else
               'Clinical parameters are largely within healthy reference ranges. Continue current healthy habits and maintain regular check-ups.'}
              </p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="hb-ring-wrap">
              <div class="hb-ring" style="color:{risk_color};">
                <div class="hb-ring-inner">{'❤️' if pct>=60 else '💚' if pct<35 else '🧡'}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Charts row ──
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                delta={'reference':50,'increasing':{'color':'#ef4444'},'decreasing':{'color':'#10b981'},
                       'font':{'size':14}},
                number={'suffix':"%",'font':{'size':40,'color':'#eef2ff','family':'Syne'}},
                gauge={'axis':{'range':[0,100],'tickfont':{'color':'#64708a','size':9},'tickcolor':'#181d30'},
                       'bar':{'color':risk_color,'thickness':0.18},
                       'bgcolor':'#0f1220','borderwidth':0,
                       'steps':[{'range':[0,25],'color':'rgba(16,185,129,0.08)'},
                                 {'range':[25,50],'color':'rgba(245,158,11,0.07)'},
                                 {'range':[50,75],'color':'rgba(230,57,80,0.08)'},
                                 {'range':[75,100],'color':'rgba(220,38,38,0.12)'}],
                       'threshold':{'line':{'color':'rgba(255,255,255,0.6)','width':2},
                                    'thickness':0.75,'value':50}},
                title={'text':f"Risk Probability — {risk_label}",'font':{'color':'#64708a','size':12}}
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                                  font_color='#dde3f5',height=240,margin=dict(l=16,r=16,t=36,b=4))
            st.plotly_chart(fig_g, use_container_width=True)

        with ch2:
            cats = ['Age','Blood Pressure','Cholesterol','Heart Rate','ST Depression','Blood Sugar']
            vals = [norm(Age,20,100), norm(RestingBP,80,200), norm(Cholesterol,100,400),
                    norm(220-MaxHR,20,160), norm(Oldpeak,0,6), float(FastingBS)]
            cc = cats+[cats[0]]; vv = vals+[vals[0]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=[0.28]*len(cc),theta=cc,fill='toself',
                fillcolor='rgba(16,185,129,0.06)',line=dict(color='#10b981',width=1,dash='dot'),name='Healthy Baseline'))
            fig_r.add_trace(go.Scatterpolar(r=vv,theta=cc,fill='toself',
                fillcolor=f'rgba({"230,57,80" if pct>=50 else "245,158,11" if pct>=35 else "16,185,129"},0.12)',
                line=dict(color=risk_color,width=2.2),name=name_d))
            fig_r.update_layout(
                polar=dict(bgcolor='#0a0d18',
                    radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color='#1e2438',size=8),
                                   gridcolor='#181d30',linecolor='#181d30'),
                    angularaxis=dict(tickfont=dict(color='#8890a4',size=10),gridcolor='#181d30',linecolor='#181d30')),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font_color='#dde3f5',height=270,margin=dict(l=40,r=40,t=28,b=12),
                legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#64708a',size=10)),
                title=dict(text="Clinical Risk Radar",font=dict(color='#64708a',size=12))
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # ════════════════════════════════════════════════════════════════
        # EXPLAINABILITY: Feature Importance + Population Percentiles
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Model Explainability</div>', unsafe_allow_html=True)

        exp1, exp2 = st.columns(2)

        with exp1:
            st.markdown("""
            <div style='font-size:13px;font-weight:600;color:#eef2ff;margin-bottom:4px;'>
              🔍 Feature Contribution Analysis
            </div>
            <div style='font-size:11px;color:var(--muted);margin-bottom:16px;'>
              Perturbation-based importance — how much each feature drives this prediction.
            </div>
            """, unsafe_allow_html=True)

            max_val = sorted_imp[0][1] if sorted_imp else 1
            for feat, val in sorted_imp:
                pct_w = (val / max_val) * 100 if max_val > 0 else 0
                bar_c = "#e63950" if pct_w > 65 else "#f59e0b" if pct_w > 35 else "#10b981"
                pct_label = f"{val*100:.1f}%"
                st.markdown(f"""
                <div class="imp-bar-wrap">
                  <div class="imp-bar-label">
                    <span>{feat}</span><span>{pct_label}</span>
                  </div>
                  <div class="imp-bar-bg">
                    <div class="imp-bar-fill" style="width:0%;background:{bar_c};"
                         data-width="{pct_w:.1f}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with exp2:
            st.markdown("""
            <div style='font-size:13px;font-weight:600;color:#eef2ff;margin-bottom:4px;'>
              📊 Population Percentile Benchmark
            </div>
            <div style='font-size:11px;color:var(--muted);margin-bottom:16px;'>
              How this patient's values compare to the 918-patient training dataset.
            </div>
            """, unsafe_allow_html=True)

            bench = [
                ("Age",            Age,        percentile_rank("Age",Age)),
                ("Resting BP",     RestingBP,  percentile_rank("RestingBP",RestingBP)),
                ("Cholesterol",    Cholesterol,percentile_rank("Cholesterol",Cholesterol)),
                ("Max Heart Rate", MaxHR,      percentile_rank("MaxHR",MaxHR)),
                ("ST Depression",  Oldpeak,    percentile_rank("Oldpeak",Oldpeak)),
            ]
            for feat, val, pctile in bench:
                if pctile is None: continue
                bc = "#e63950" if pctile>75 else "#f59e0b" if pctile>50 else "#10b981"
                st.markdown(f"""
                <div class="pct-wrap">
                  <div class="pct-name">{feat}</div>
                  <div class="pct-bar-outer">
                    <div class="pct-bar-inner" style="width:{pctile}%;background:{bc};transition:width 1.2s ease;"></div>
                  </div>
                  <div class="pct-val" style="color:{bc};">{pctile:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size:10px;color:var(--muted);margin-top:10px;'>
            Percentile = higher than X% of training population. High percentile on risk factors = higher risk.
            </div>
            """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # WHAT-IF SIMULATOR
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">What-If Simulator</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="whatif-card">
          <div class="whatif-title">🧪 Simulate Clinical Improvements</div>
          <div class="whatif-sub">Adjust values below to see how lifestyle or treatment changes could affect your risk score.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        w1,w2,w3,w4 = st.columns(4)
        with w1: wi_bp  = st.slider("Resting BP (mmHg)", 80, 200, RestingBP, key="wi_bp")
        with w2: wi_chol= st.slider("Cholesterol (mg/dL)", 100, 400, min(Cholesterol,400), key="wi_chol")
        with w3: wi_hr  = st.slider("Max Heart Rate (bpm)", 60, 220, MaxHR, key="wi_hr")
        with w4: wi_op  = st.slider("ST Depression", 0.0, 6.0, float(max(Oldpeak,0.0)), key="wi_op", step=0.1)

        # Build what-if dataframe
        wi_df = input_df.copy()
        wi_df['RestingBP'] = wi_bp
        wi_df['Cholesterol'] = wi_chol
        wi_df['MaxHR'] = wi_hr
        wi_df['Oldpeak'] = wi_op

        wi_pred = model.predict(wi_df)[0]
        try:    wi_prob = model.predict_proba(wi_df)[0][1]
        except: wi_prob = 0.72 if wi_pred==1 else 0.28
        wi_pct = round(wi_prob*100, 1)
        wi_delta = pct - wi_pct
        wi_lbl, wi_col, wi_icon = stratify_risk(wi_pct)

        wc1,wc2,wc3 = st.columns(3)
        with wc1:
            st.markdown(f"""
            <div style='background:var(--surface2);border:1px solid var(--border2);border-radius:12px;padding:18px;text-align:center;'>
              <div style='font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Current Risk</div>
              <div style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:{risk_color};'>{pct}%</div>
              <div style='font-size:11px;color:var(--muted);'>{risk_label}</div>
            </div>""", unsafe_allow_html=True)
        with wc2:
            arrow = "↓" if wi_delta>0 else "↑" if wi_delta<0 else "→"
            delta_col = "#10b981" if wi_delta>0 else "#e63950" if wi_delta<0 else "#64708a"
            st.markdown(f"""
            <div style='background:var(--surface2);border:1px solid var(--border2);border-radius:12px;padding:18px;text-align:center;'>
              <div style='font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Change</div>
              <div style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:{delta_col};'>{arrow} {abs(wi_delta):.1f}%</div>
              <div style='font-size:11px;color:var(--muted);'>{'Improvement' if wi_delta>0 else 'Worsening' if wi_delta<0 else 'No change'}</div>
            </div>""", unsafe_allow_html=True)
        with wc3:
            st.markdown(f"""
            <div style='background:var(--surface2);border:1px solid var(--border2);border-radius:12px;padding:18px;text-align:center;'>
              <div style='font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Simulated Risk</div>
              <div style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:{wi_col};'>{wi_pct}%</div>
              <div style='font-size:11px;color:var(--muted);'>{wi_lbl}</div>
            </div>""", unsafe_allow_html=True)

        # Comparison bar
        fig_wi = go.Figure()
        fig_wi.add_trace(go.Bar(name='Current', x=['Risk Score'], y=[pct],
            marker_color=risk_color, width=0.25, text=[f"{pct}%"], textposition='outside',
            textfont=dict(color='#64708a')))
        fig_wi.add_trace(go.Bar(name='Simulated', x=['Risk Score'], y=[wi_pct],
            marker_color=wi_col, width=0.25, text=[f"{wi_pct}%"], textposition='outside',
            textfont=dict(color='#64708a')))
        fig_wi.add_hline(y=50, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'))
        fig_wi.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#dde3f5',family='Instrument Sans'),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(range=[0,115], gridcolor='#181d30', title='Risk %'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#64708a')),
            height=220, margin=dict(l=8,r=8,t=16,b=8)
        )
        st.plotly_chart(fig_wi, use_container_width=True)

        # ════════════════════════════════════════════════════════════════
        # RECOMMENDATIONS
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Personalised Recommendations</div>', unsafe_allow_html=True)

        if pct >= 60:
            tips = [
                ("🩺","Cardiology Consultant","Schedule a full cardiac workup: stress ECG, echocardiogram, and coronary angiography as advised."),
                ("💊","Medication Review","Discuss statin therapy, antihypertensives, or antiplatelet agents with your cardiologist."),
                ("🥗","Therapeutic Diet","DASH or Mediterranean diet. Sodium < 2g/day, reduce saturated fats and refined carbs."),
                ("🏃","Supervised Exercise","Cardiac rehab or medically supervised low-impact exercise 30 min, 5× per week."),
                ("📊","Quarterly Monitoring","Track BP, lipids, and fasting glucose every 3 months. Monitor resting HR with a wearable."),
                ("🚭","Lifestyle Overhaul","Quit smoking, limit alcohol to 1 unit/day, address obesity, manage stress professionally."),
            ]
        elif pct >= 35:
            tips = [
                ("🩺","Medical Review","Annual cardiac check-up and lipid screening. Discuss risk factors with your GP."),
                ("🥦","Improve Nutrition","Reduce processed foods, added sugars, and trans fats. Increase fibre, vegetables, and omega-3."),
                ("🏃","Regular Exercise","150+ min/week of moderate cardio. Even brisk walking significantly lowers cardiovascular risk."),
                ("🧘","Stress Reduction","Chronic stress drives cortisol and BP. Try mindfulness, yoga, or therapy."),
                ("⚖️","Weight Management","Even a 5–10% weight loss meaningfully reduces BP, cholesterol, and heart disease risk."),
                ("😴","Sleep Quality","7–9 hours nightly. Poor sleep is an independent risk factor for cardiovascular disease."),
            ]
        else:
            tips = [
                ("💪","Keep Active","150+ min/week of moderate exercise. Mix aerobic and resistance training for long-term cardiac resilience."),
                ("🥦","Eat Well","Whole foods, lean proteins, healthy fats. High fibre and low processed food diet."),
                ("😴","Prioritise Sleep","7–9 hours nightly. Good sleep lowers inflammation and cortisol — both heart disease precursors."),
                ("🧘","Manage Stress","Even low-risk individuals benefit from stress management. Mindfulness reduces long-term BP."),
                ("📅","Annual Screening","Lipid panel, BP, and fasting glucose once a year. Prevention starts with awareness."),
                ("💧","Stay Hydrated","2–3L of water daily supports blood viscosity and optimal cardiac output."),
            ]

        st.markdown('<div class="tip-grid">'
            + ''.join(f'<div class="tip-card" style="animation-delay:{0.05*i}s">'
                      f'<div class="tip-icon">{ic}</div>'
                      f'<div class="tip-title">{t}</div>'
                      f'<div class="tip-body">{b}</div></div>'
                      for i,(ic,t,b) in enumerate(tips))
            + '</div>', unsafe_allow_html=True)

        # ── Download Report ──
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         CARDIOCARE AI — FULL ASSESSMENT REPORT               ║
╚══════════════════════════════════════════════════════════════╝

Patient Name   : {entry['name']}
Date / Time    : {datetime.now().strftime('%d %B %Y, %H:%M')}
──────────────────────────────────────────────────────────────

RESULT         : {risk_icon} {risk_label.upper()} RISK  ({pct}%)
Cardiac Age    : {health_age} yrs  (Actual Age: {Age})
BMI            : {s1.get('bmi',0):.1f}  ({s1.get('bmi_cat','')})

RISK STRATIFICATION:
  Low Risk     : 0–24%
  Moderate Risk: 25–49%
  High Risk    : 50–74%
  Critical Risk: 75–100%
  This Patient : {pct}% → {risk_label}

──────────────────────────────────────────────────────────────
CLINICAL PARAMETERS
──────────────────────────────────────────────────────────────
Age              : {Age} years
Sex              : {gender}
Resting BP       : {RestingBP} mmHg
Cholesterol      : {Cholesterol} mg/dL
Fasting Blood Sugar : {'High (>120 mg/dL)' if FastingBS else 'Normal'}
Max Heart Rate   : {MaxHR} bpm
Chest Pain Type  : {ChestPainType}
Resting ECG      : {RestingECG}
Exercise Angina  : {'Yes' if ExerciseAngina=='Y' else 'No'}
ST Depression    : {Oldpeak}
ST Slope         : {ST_Slope}

──────────────────────────────────────────────────────────────
TOP CONTRIBUTING FEATURES (Perturbation Analysis)
──────────────────────────────────────────────────────────────
{chr(10).join(f"  {i+1}. {f:<22} {v*100:.1f}% contribution" for i,(f,v) in enumerate(sorted_imp))}

──────────────────────────────────────────────────────────────
DISCLAIMER: For educational use only. Not a clinical tool.
Consult a qualified cardiologist for medical advice.
──────────────────────────────────────────────────────────────
Generated by CardioCare AI | SVM Model | Final Year Project
"""
        b64r = base64.b64encode(report.encode()).decode()
        fname = f"CardioCare_{entry['name'].replace(' ','_')}.txt"
        st.markdown(f'<a href="data:file/txt;base64,{b64r}" download="{fname}" class="dl-btn">📄 Download Full Assessment Report</a>',
                    unsafe_allow_html=True)

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        _,rb = st.columns([3,1])
        with rb:
            if st.button("🔄 New Assessment", key="restart"):
                st.session_state.step=1
                st.session_state.predicted=False
                st.session_state.patient_name=""
                for k in ['s1','s2','s3']: st.session_state.pop(k,None)
                st.rerun()

        st.info("⚠️ **Clinical Disclaimer:** This tool is for educational and academic purposes only. It does not constitute medical advice, diagnosis, or treatment. Consult a qualified physician.")

    st.markdown('</div>', unsafe_allow_html=True)  # page-body

# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in nav:
    st.markdown('<div class="page-body">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#eef2ff;margin-bottom:5px;'>Session History</div>
    <div style='font-size:12px;color:#64708a;margin-bottom:24px;'>All assessments conducted in this session.</div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No assessments yet. Complete a patient assessment first.")
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            rl, rc2, ri = stratify_risk(h['prob'])
            badge_cls = "badge-r" if h['pred']==1 and h['prob']>=60 else "badge-a" if h['prob']>=35 else "badge-g"
            st.markdown(f"""
            <div class="hist-row">
              <div style='font-weight:700;font-size:14px;flex:1;'>#{len(st.session_state.history)-i} — {h['name']}</div>
              <div style='font-size:12px;color:#64708a;'>Age {h['age']} · {h['gender']}</div>
              <span class="badge {badge_cls}">{ri} {rl}</span>
              <div style='font-size:13px;font-weight:700;color:{rc2};'>{h['prob']}%</div>
              <div style='font-size:12px;color:#64708a;'>♥ Age: <strong style='color:#eef2ff;'>{h['health_age']}</strong></div>
              <div style='font-size:12px;color:#64708a;'>BMI {h.get('bmi',0):.1f} · {h.get('bmi_cat','')}</div>
              <div style='font-size:11px;color:#64708a;'>{h['time']}</div>
            </div>""", unsafe_allow_html=True)

        if len(st.session_state.history) > 1:
            st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Multi-Patient Risk Comparison</div>', unsafe_allow_html=True)
            names  = [h['name'] or f"P{i+1}" for i,h in enumerate(st.session_state.history)]
            probs  = [h['prob'] for h in st.session_state.history]
            h_ages = [h['health_age'] for h in st.session_state.history]
            ages   = [h['age'] for h in st.session_state.history]
            cols2  = [stratify_risk(p)[1] for p in probs]

            fc1, fc2 = st.columns(2)
            with fc1:
                fig_cmp = go.Figure(go.Bar(x=names,y=probs,marker=dict(color=cols2,line=dict(width=0)),
                    text=[f"{p}%" for p in probs],textposition='outside',textfont=dict(color='#64708a')))
                fig_cmp.add_hline(y=50,line=dict(color='rgba(255,255,255,0.2)',width=1,dash='dot'))
                fig_cmp.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#dde3f5',family='Instrument Sans'),
                    xaxis=dict(gridcolor='#181d30',linecolor='#181d30'),
                    yaxis=dict(title='Risk Score (%)',gridcolor='#181d30',range=[0,118]),
                    title=dict(text='Risk Scores',font=dict(color='#64708a',size=13)),
                    height=280,margin=dict(l=8,r=8,t=36,b=8))
                st.plotly_chart(fig_cmp, use_container_width=True)

            with fc2:
                fig_age = go.Figure()
                fig_age.add_trace(go.Scatter(x=names,y=ages,name='Actual Age',
                    line=dict(color='#60a5fa',width=2),marker=dict(size=7)))
                fig_age.add_trace(go.Scatter(x=names,y=h_ages,name='Cardiac Age',
                    line=dict(color='#e63950',width=2,dash='dot'),marker=dict(size=7)))
                fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#dde3f5',family='Instrument Sans'),
                    xaxis=dict(gridcolor='#181d30',linecolor='#181d30'),
                    yaxis=dict(title='Age (years)',gridcolor='#181d30'),
                    legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#64708a',size=11)),
                    title=dict(text='Actual vs Cardiac Age',font=dict(color='#64708a',size=13)),
                    height=280,margin=dict(l=8,r=8,t=36,b=8))
                st.plotly_chart(fig_age, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif "Methodology" in nav:
    st.markdown('<div class="page-body">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#eef2ff;margin-bottom:5px;'>Methodology</div>
    <div style='font-size:12px;color:#64708a;margin-bottom:24px;'>Technical documentation — dataset, model, features, and explainability approach.</div>
    """, unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📊 Dataset","🤖 Model","📐 Features","🔬 Explainability"])

    with t1:
        st.markdown("""
**Heart Failure Prediction Dataset** — UCI ML Repository  
Aggregated from 5 independent medical institutions:

| Institution | Location | Records |
|-------------|----------|---------|
| Cleveland Clinic Foundation | USA | 303 |
| Hungarian Institute of Cardiology | Hungary | 294 |
| University Hospitals of Geneva | Switzerland | 123 |
| Long Beach VA Medical Center | USA | 200 |
| Stalog (Heart) Dataset | — | ~270 |

**Total: 918 records** · 11 features · Binary target (0 = No Disease, 1 = Disease)  
Class distribution: ~55% positive, ~45% negative — well-balanced for binary classification.
        """)

    with t2:
        st.markdown("""
**Algorithm:** Support Vector Machine (SVM) with RBF Kernel

| Pipeline Step | Detail |
|--------------|--------|
| Preprocessing | StandardScaler on numerics; one-hot encoding for categoricals |
| Kernel | Radial Basis Function — handles non-linear boundaries |
| Tuning | GridSearchCV, 5-fold stratified cross-validation on C and gamma |
| Split | 80/20 train/test (stratified) |

**Test Set Performance:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | ~87% | Overall correct predictions |
| Precision | ~88% | Of predicted positives, how many are true positives |
| Recall | ~90% | Of actual positives, how many were caught (critical in medicine) |
| F1-Score | ~89% | Harmonic mean of precision & recall |
| ROC-AUC | ~93% | Discrimination ability across all thresholds |

High recall is prioritised: in clinical settings, missing a true positive (false negative) is far more dangerous than a false alarm.
        """)

    with t3:
        st.markdown("""
| Feature | Type | Reference Range | Clinical Significance |
|---------|------|----------------|-----------------------|
| Age | Numerical | — | Risk increases with age |
| Sex | Binary | — | Males have higher baseline CVD risk |
| ChestPainType | Categorical | — | ASY = asymptomatic, paradoxically highest risk |
| RestingBP | Numerical | < 120 mmHg | Sustained hypertension damages arterial walls |
| Cholesterol | Numerical | < 200 mg/dL | LDL deposits contribute to plaque formation |
| FastingBS | Binary | ≤ 120 mg/dL | Diabetes is a major CVD risk multiplier |
| RestingECG | Categorical | Normal | LVH and ST changes indicate structural problems |
| MaxHR | Numerical | 100–170 bpm | Lower peak HR indicates poor cardiac reserve |
| ExerciseAngina | Binary | No | Angina during exertion = impaired coronary flow |
| Oldpeak | Numerical | 0–1.0 | ST depression > 2 is clinically significant |
| ST_Slope | Categorical | Upsloping | Downsloping = most concerning pattern |
        """)

    with t4:
        st.markdown("""
**Explainability Approach: Model-Agnostic Perturbation Analysis**

Unlike black-box interpretations, this system uses a transparent, model-agnostic method to explain predictions:

1. **Baseline probability** is computed from the original input
2. **Each feature is individually zeroed out** (perturbed to its baseline/null value)
3. The **change in predicted probability** is measured
4. Larger change = higher importance for that prediction

This is similar in concept to SHAP (SHapley Additive exPlanations) but requires no external dependencies, making it portable and fully transparent.

**Advantages:**
- Works with ANY sklearn-compatible model
- Patient-specific: explains THIS prediction, not a global average
- No additional library dependencies (no shap, no lime)
- Results are intuitive: "removing X changed the prediction by Y%"

**What-If Simulator** extends this by allowing interactive manipulation of key modifiable risk factors (BP, cholesterol, heart rate, ST depression) to simulate the effect of clinical interventions or lifestyle changes on predicted risk.
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div><strong>CardioCare AI v2.0</strong> · Final Year Capstone · Department of Computer Science</div>
    <div>SVM · RBF Kernel · UCI Heart Failure Dataset (918) · Perturbation Explainability · Educational Use Only</div>
</div>
""", unsafe_allow_html=True)