/**
 * TrollFish — report-pdf.js
 * Generates A4 PDF reports using jsPDF + Canvas-based chart rendering.
 * Faithful port of Python report_pdf.py — same layout, colours, heatmaps,
 * histograms, maps, tables and gold dots.
 *
 * Dependencies (loaded from CDN on first use):
 *   - jspdf (UMD)
 *   - jspdf-autotable
 *
 * Main entry:  generatePdf(reportData, opts) → Blob
 */

import { PALETTE } from './config.js';

// ── CDN / library loading ─────────────────────────────────────────────

const JSPDF_CDN = 'https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js';
const AUTOTABLE_CDN = 'https://cdn.jsdelivr.net/npm/jspdf-autotable@3.8.4/dist/jspdf.plugin.autotable.min.js';

let _jsPDF = null;
const _imageLoadCache = new Map();

async function loadJsPdf() {
  if (_jsPDF) return _jsPDF;
  await loadScript(JSPDF_CDN);
  await loadScript(AUTOTABLE_CDN);
  _jsPDF = window.jspdf.jsPDF;
  if (!_jsPDF) throw new Error('jsPDF failed to load');
  return _jsPDF;
}

function loadScript(url) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${url}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = url;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(s);
  });
}

// ── Colour helpers (matches Python PALETTE + helpers) ─────────────────

function hexToRgb(hex) {
  const m = hex.replace('#', '');
  return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
}

function athleteColor(idx) {
  return PALETTE[idx % PALETTE.length];
}

function validAthleteColor(color) {
  return typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color.trim());
}

function darken(rgb, factor = 0.7) {
  return rgb.map(c => Math.max(Math.round(c * factor), 0));
}

const TABLE_HEADER_BG = [41, 128, 185];
const TABLE_HEADER_FG = [255, 255, 255];
const TABLE_ALT_BG = [240, 245, 250];
const TABLE_METRIC_BG = [228, 228, 228];
const TABLE_METRIC_HEADER_BG = [110, 110, 110];
const GOLD_COLOR = [255, 200, 0];
const CANVAS_FONT_SCALE = 1.18;
const PDF_FONT_SCALE = 1.08;
const PDF_HEATMAP_YIELD_BUDGET_MS = 12;
const PDF_HEATMAP_KERNEL_CUTOFF_SIGMA = 3.5;

// Fixed axis limits for histograms — enables cross-session comparison
const HISTOGRAM_XLIMITS = {
  'Trunk Angle':       [0, 120],
  'Rudder Angle':      [-90, 90],
  'Boom Angle':        [-180, 180],
  'Rolling Moment':    [0, 800],
  'Heel Angle':        [-45, 45],
  'Speed Over Ground': [0, 18],
};

function heatmapNowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') return performance.now();
  return Date.now();
}

function yieldHeatmapWork() {
  return new Promise(resolve => {
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(resolve, 0);
  });
}

function createHeatmapYieldController(budgetMs = PDF_HEATMAP_YIELD_BUDGET_MS) {
  let lastYieldAt = heatmapNowMs();
  return {
    async maybeYield(force = false) {
      const now = heatmapNowMs();
      if (!force && (now - lastYieldAt) < budgetMs) return;
      await yieldHeatmapWork();
      lastYieldAt = heatmapNowMs();
    },
  };
}

// ── Gaussian KDE (mirrors scipy gaussian_kde with Scott's rule) ───────

function gaussianKde(data, xs, bandwidth = null) {
  const n = data.length;
  if (n < 2) return xs.map(() => 0);
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(data.reduce((a, v) => a + (v - mean) ** 2, 0) / n);
  const h = bandwidth || (std * Math.pow(n, -0.2));
  if (h === 0) return xs.map(() => 0);
  const invH = 1 / h;
  const norm = 1 / (n * h * Math.sqrt(2 * Math.PI));
  return xs.map(x => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const u = (x - data[i]) * invH;
      sum += Math.exp(-0.5 * u * u);
    }
    return sum * norm;
  });
}

// 2D Gaussian KDE on a grid (mirrors scipy gaussian_kde for heatmaps)
function prepareGaussianKde2d(pointsXY, xGrid, yGrid, bandwidth = 0.3) {
  const n = pointsXY.length;
  if (n < 3) return null;

  // Compute per-axis std for bandwidth scaling
  let sx = 0, sy = 0, mx = 0, my = 0;
  for (const [x, y] of pointsXY) { mx += x; my += y; }
  mx /= n; my /= n;
  for (const [x, y] of pointsXY) { sx += (x - mx) ** 2; sy += (y - my) ** 2; }
  sx = Math.sqrt(sx / n); sy = Math.sqrt(sy / n);

  // Scott's rule adjusted by bandwidth param
  const hx = (sx || 0.5) * bandwidth * Math.pow(n, -0.2);
  const hy = (sy || 0.5) * bandwidth * Math.pow(n, -0.2);
  if (hx === 0 || hy === 0) return null;

  const nx = xGrid.length;
  const ny = yGrid.length;
  if (nx === 0 || ny === 0) return null;

  const xStep = nx > 1 ? (xGrid[1] - xGrid[0]) : 1;
  const yStep = ny > 1 ? (yGrid[1] - yGrid[0]) : 1;
  if (!Number.isFinite(xStep) || !Number.isFinite(yStep) || xStep === 0 || yStep === 0) return null;

  return {
    n,
    nx,
    ny,
    hx,
    hy,
    invHx2: 1 / (2 * hx * hx),
    invHy2: 1 / (2 * hy * hy),
    x0: xGrid[0],
    y0: yGrid[0],
    xStep,
    yStep,
    xGrid,
    yGrid,
  };
}

function normalizeKdeGridInPlace(Z) {
  let maxZ = 0;
  for (let i = 0; i < Z.length; i++) if (Z[i] > maxZ) maxZ = Z[i];
  if (maxZ > 0) {
    for (let i = 0; i < Z.length; i++) Z[i] /= maxZ;
  }
  return maxZ;
}

function gaussianKde2d(pointsXY, xGrid, yGrid, bandwidth = 0.3) {
  const prep = prepareGaussianKde2d(pointsXY, xGrid, yGrid, bandwidth);
  if (!prep) return null;

  const { nx, ny, x0, y0, xStep, yStep, invHx2, invHy2, hx, hy } = prep;
  const Z = new Float32Array(nx * ny);
  const xCut = hx * PDF_HEATMAP_KERNEL_CUTOFF_SIGMA;
  const yCut = hy * PDF_HEATMAP_KERNEL_CUTOFF_SIGMA;

  for (let pi = 0; pi < pointsXY.length; pi++) {
    const px = pointsXY[pi][0];
    const py = pointsXY[pi][1];
    const ixStart = Math.max(0, Math.ceil(((px - xCut) - x0) / xStep));
    const ixEnd = Math.min(nx - 1, Math.floor(((px + xCut) - x0) / xStep));
    const iyStart = Math.max(0, Math.ceil(((py - yCut) - y0) / yStep));
    const iyEnd = Math.min(ny - 1, Math.floor(((py + yCut) - y0) / yStep));
    if (ixStart > ixEnd || iyStart > iyEnd) continue;

    const xWeights = new Float32Array(ixEnd - ixStart + 1);
    for (let ix = ixStart; ix <= ixEnd; ix++) {
      const dx = xGrid[ix] - px;
      xWeights[ix - ixStart] = Math.exp(-(dx * dx) * invHx2);
    }

    for (let iy = iyStart; iy <= iyEnd; iy++) {
      const dy = yGrid[iy] - py;
      const yWeight = Math.exp(-(dy * dy) * invHy2);
      const rowBase = iy * nx + ixStart;
      for (let xIdx = 0; xIdx < xWeights.length; xIdx++) {
        Z[rowBase + xIdx] += xWeights[xIdx] * yWeight;
      }
    }
  }

  normalizeKdeGridInPlace(Z);
  return { Z, nx, ny };
}

async function gaussianKde2dAsync(pointsXY, xGrid, yGrid, bandwidth = 0.3, yieldController = null) {
  const prep = prepareGaussianKde2d(pointsXY, xGrid, yGrid, bandwidth);
  if (!prep) return null;

  const { n, nx, ny, x0, y0, xStep, yStep, invHx2, invHy2, hx, hy } = prep;
  const Z = new Float32Array(nx * ny);
  const xCut = hx * PDF_HEATMAP_KERNEL_CUTOFF_SIGMA;
  const yCut = hy * PDF_HEATMAP_KERNEL_CUTOFF_SIGMA;

  for (let pi = 0; pi < n; pi++) {
    const px = pointsXY[pi][0];
    const py = pointsXY[pi][1];
    const ixStart = Math.max(0, Math.ceil(((px - xCut) - x0) / xStep));
    const ixEnd = Math.min(nx - 1, Math.floor(((px + xCut) - x0) / xStep));
    const iyStart = Math.max(0, Math.ceil(((py - yCut) - y0) / yStep));
    const iyEnd = Math.min(ny - 1, Math.floor(((py + yCut) - y0) / yStep));
    if (ixStart > ixEnd || iyStart > iyEnd) {
      if (yieldController && (pi % 24) === 23) await yieldController.maybeYield();
      continue;
    }

    const xWeights = new Float32Array(ixEnd - ixStart + 1);
    for (let ix = ixStart; ix <= ixEnd; ix++) {
      const dx = xGrid[ix] - px;
      xWeights[ix - ixStart] = Math.exp(-(dx * dx) * invHx2);
    }

    for (let iy = iyStart; iy <= iyEnd; iy++) {
      const dy = yGrid[iy] - py;
      const yWeight = Math.exp(-(dy * dy) * invHy2);
      const rowBase = iy * nx + ixStart;
      for (let xIdx = 0; xIdx < xWeights.length; xIdx++) {
        Z[rowBase + xIdx] += xWeights[xIdx] * yWeight;
      }
      if (yieldController && ((iy - iyStart) % 10) === 9) await yieldController.maybeYield();
    }
    if (yieldController && (pi % 8) === 7) await yieldController.maybeYield();
  }

  let maxZ = 0;
  for (let i = 0; i < Z.length; i++) {
    if (Z[i] > maxZ) maxZ = Z[i];
    if (yieldController && (i % 4096) === 4095) await yieldController.maybeYield();
  }
  if (maxZ > 0) {
    for (let i = 0; i < Z.length; i++) {
      Z[i] /= maxZ;
      if (yieldController && (i % 4096) === 4095) await yieldController.maybeYield();
    }
  }

  return { Z, nx, ny };
}

// ── Canvas helpers ────────────────────────────────────────────────────

function createCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const rawCtx = c.getContext('2d');
  const ctx = new Proxy(rawCtx, {
    get(target, prop) {
      const value = target[prop];
      return typeof value === 'function' ? value.bind(target) : value;
    },
    set(target, prop, value) {
      if (prop === 'font' && typeof value === 'string') {
        value = value.replace(/(\d+(?:\.\d+)?)px/g, (m, n) => `${(parseFloat(n) * CANVAS_FONT_SCALE).toFixed(2).replace(/\.?0+$/, '')}px`);
      }
      target[prop] = value;
      return true;
    },
  });
  return { canvas: c, ctx };
}

function encodeCanvasJpeg(canvas, quality = 0.82) {
  return canvas.toDataURL('image/jpeg', quality);
}

function encodeCanvasPng(canvas) {
  return canvas.toDataURL('image/png');
}

function addImageDataUrl(pdf, dataUrl, x, y, w, h) {
  const fmt = String(dataUrl || '').startsWith('data:image/jpeg') ? 'JPEG' : 'PNG';
  const compression = fmt === 'JPEG' ? 'MEDIUM' : 'FAST';
  pdf.addImage(dataUrl, fmt, x, y, w, h, undefined, compression);
}

function loadImage(src) {
  if (_imageLoadCache.has(src)) return _imageLoadCache.get(src);
  const promise = new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous'; // needed for CartoDB tiles → canvas toDataURL
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = src;
  }).catch(err => {
    _imageLoadCache.delete(src);
    throw err;
  });
  _imageLoadCache.set(src, promise);
  return promise;
}

// ── KDE Histogram (matches Python _make_smooth_histogram) ─────────────

export function renderKdeHistogram(dataByAthlete, athleteColors, xlabel, title = '') {
  const W = 660, H = 360;
  const PAD = { top: 30, right: 20, bottom: 45, left: 55 };
  const { canvas, ctx } = createCanvas(W, H);
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Determine x-range (fixed limits when defined, like Python)
  const fixedLimits = HISTOGRAM_XLIMITS[xlabel];
  let xMin, xMax;
  const allVals = [];
  for (const { values } of dataByAthlete) allVals.push(...values);
  if (fixedLimits) {
    [xMin, xMax] = fixedLimits;
  } else if (allVals.length > 0) {
    const mn = Math.min(...allVals);
    const mx = Math.max(...allVals);
    const margin = (mx - mn) * 0.15 || 1;
    xMin = mn - margin;
    xMax = mx + margin;
  } else {
    xMin = 0; xMax = 1;
  }

  // Compute KDE for each athlete (300 points like Python)
  const nPts = 300;
  const xs = Array.from({ length: nPts }, (_, i) => xMin + (xMax - xMin) * i / (nPts - 1));
  const curves = [];
  let yGlobalMax = 0;
  for (const { values, label, color } of dataByAthlete) {
    if (values.length < 3) continue;
    const ys = gaussianKde(values, xs);
    const ym = Math.max(...ys);
    if (ym > yGlobalMax) yGlobalMax = ym;
    curves.push({ xs, ys, label, color, values });
  }
  if (yGlobalMax === 0) yGlobalMax = 1;

  // Grid (like Python alpha=0.3)
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.globalAlpha = 0.3;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const x = PAD.left + (pw * i) / 5;
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + ph); ctx.stroke();
  }
  ctx.globalAlpha = 1.0;

  // Draw curves (matching Python: fill_between alpha=0.25, line lw=1.5)
  const meanMarkers = [];
  for (const curve of curves) {
    const rgb = hexToRgb(curve.color);

    // Fill area (alpha=0.25 like Python)
    ctx.fillStyle = `rgba(${rgb.join(',')}, 0.25)`;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + ph);
    for (let i = 0; i < nPts; i++) {
      const x = PAD.left + ((curve.xs[i] - xMin) / (xMax - xMin)) * pw;
      const y = PAD.top + ph - (curve.ys[i] / yGlobalMax) * ph;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(PAD.left + pw, PAD.top + ph);
    ctx.closePath();
    ctx.fill();

    // KDE line (linewidth 1.5 like Python)
    ctx.strokeStyle = curve.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) {
      const x = PAD.left + ((curve.xs[i] - xMin) / (xMax - xMin)) * pw;
      const y = PAD.top + ph - (curve.ys[i] / yGlobalMax) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Mean vertical dashed line (Python: linewidth=1.2, linestyle="--", alpha=0.8)
    const mean = curve.values.reduce((a, b) => a + b, 0) / curve.values.length;
    const meanX = PAD.left + ((mean - xMin) / (xMax - xMin)) * pw;
    ctx.strokeStyle = curve.color;
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.8;
    ctx.setLineDash([5, 3]);
    ctx.beginPath(); ctx.moveTo(meanX, PAD.top); ctx.lineTo(meanX, PAD.top + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
    meanMarkers.push({ x: meanX, color: curve.color, text: mean.toFixed(1) });
  }

  // Mean labels with collision-aware vertical stacking.
  ctx.font = '10px sans-serif';
  const placed = [];
  const sorted = [...meanMarkers].sort((a, b) => a.x - b.x);
  for (const m of sorted) {
    const textW = ctx.measureText(m.text).width;
    const boxW = textW + 8;
    const boxH = 14;
    const boxX = m.x - boxW / 2;
    let boxY = PAD.top + 4;

    while (
      placed.some(p => !(boxX + boxW < p.x || boxX > p.x + p.w || boxY + boxH < p.y || boxY > p.y + p.h))
      && boxY + boxH < PAD.top + ph - 2
    ) {
      boxY += boxH + 3;
    }

    placed.push({ x: boxX, y: boxY, w: boxW, h: boxH });

    ctx.globalAlpha = 0.7;
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 0.5;
    _roundRect(ctx, boxX, boxY, boxW, boxH, 3);
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    ctx.fillStyle = m.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(m.text, m.x, boxY + boxH / 2);
  }

  // Axes (solid black)
  ctx.strokeStyle = '#222222';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + ph);
  ctx.lineTo(PAD.left + pw, PAD.top + ph);
  ctx.stroke();

  // X-axis tick labels
  ctx.fillStyle = '#222222';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 5; i++) {
    const val = xMin + (xMax - xMin) * i / 5;
    const x = PAD.left + (pw * i) / 5;
    ctx.fillText(val.toFixed(1), x, PAD.top + ph + 5);
  }

  // Y-axis tick labels
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const val = yGlobalMax * (4 - i) / 4;
    const y = PAD.top + (ph * i) / 4;
    ctx.fillText(val.toFixed(3), PAD.left - 5, y);
  }

  // Title (Python: fontsize=10, fontweight="bold")
  if (title) {
    ctx.fillStyle = '#222222';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(title, W / 2, 4);
  }

  // X-axis label (Python: fontsize=9)
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(xlabel, PAD.left + pw / 2, H - 2);

  // Y-axis label
  ctx.save();
  ctx.font = '11px sans-serif';
  ctx.fillStyle = '#222222';
  ctx.textAlign = 'center';
  ctx.translate(12, PAD.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Density', 0, 0);
  ctx.restore();

  // Legend (Python: upper right, fontsize=7) — always show so athlete name+color is visible
  if (curves.length >= 1) {
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let ly = PAD.top + 8;
    for (const c of curves) {
      ctx.fillStyle = c.color;
      ctx.fillRect(PAD.left + pw - 90, ly, 10, 10);
      ctx.fillStyle = '#222222';
      ctx.fillText(c.label, PAD.left + pw - 76, ly + 5);
      ly += 14;
    }
  }

  return encodeCanvasPng(canvas);
}

function _roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

// ── Mercator projection ───────────────────────────────────────────────

function countPolarCurveBins(polar) {
  return (Array.isArray(polar?.points) ? polar.points : []).filter((point) => {
    const twaDeg = Number(point?.twaDeg);
    const athleteSpeedKts = Number(point?.athleteSpeedKts);
    const sampleCount = Number(point?.sampleCount);
    return Number.isFinite(twaDeg) && Number.isFinite(athleteSpeedKts) && athleteSpeedKts >= 0 && sampleCount > 0;
  }).length;
}

function hasPolarCurveData(polar) {
  return countPolarCurveBins(polar) >= 2;
}

function polarToCanvasPoint(twaDeg, speedKts, speedCeiling, tackSign, centerX, centerY, radius) {
  const angleRad = (tackSign < 0 ? -twaDeg : twaDeg) * Math.PI / 180;
  const dist = radius * Math.max(0, Math.min(speedKts / Math.max(speedCeiling, 1e-6), 1.25));
  return {
    x: centerX + Math.sin(angleRad) * dist,
    y: centerY - Math.cos(angleRad) * dist,
  };
}

function buildPolarCurvePath(rows, speedField, speedCeiling, centerX, centerY, radius) {
  const filtered = rows
    .filter((row) => {
      const twaDeg = Number(row?.twaDeg);
      const speedKts = Number(row?.[speedField]);
      return Number.isFinite(twaDeg) && Number.isFinite(speedKts) && speedKts >= 0;
    })
    .sort((a, b) => Number(a?.twaDeg || 0) - Number(b?.twaDeg || 0));
  if (filtered.length < 2) return [];

  const port = filtered
    .slice()
    .reverse()
    .map((row) => polarToCanvasPoint(Number(row.twaDeg), Number(row[speedField]), speedCeiling, -1, centerX, centerY, radius));
  const starboard = filtered
    .map((row) => polarToCanvasPoint(Number(row.twaDeg), Number(row[speedField]), speedCeiling, 1, centerX, centerY, radius));
  return port.concat(starboard);
}

function strokePolarCurve(ctx, points, color, lineWidth = 3) {
  if (!Array.isArray(points) || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    if (i === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  }
  ctx.stroke();
  ctx.restore();
}

function renderPolarPlot(polar, athleteColor = '#d95d39') {
  if (!hasPolarCurveData(polar)) return null;

  const W = 700;
  const H = 700;
  const { canvas, ctx } = createCanvas(W, H);
  const centerX = W / 2;
  const centerY = H / 2 + 18;
  const radius = 252;
  const athleteStroke = validAthleteColor(athleteColor) ? athleteColor.trim() : '#d95d39';
  const athleteRows = (Array.isArray(polar?.points) ? polar.points : [])
    .filter((point) =>
      Number.isFinite(Number(point?.twaDeg)) &&
      Number(point?.sampleCount) > 0 &&
      Number.isFinite(Number(point?.athleteSpeedKts)),
    )
    .sort((a, b) => Number(a?.twaDeg || 0) - Number(b?.twaDeg || 0));

  let maxSpeed = 0;
  for (const row of athleteRows) {
    const athleteSpeed = Number(row?.athleteSpeedKts);
    if (Number(row?.sampleCount) > 0 && Number.isFinite(athleteSpeed)) maxSpeed = Math.max(maxSpeed, athleteSpeed);
  }
  const ringStep = maxSpeed <= 4 ? 0.5 : 1.0;
  const speedCeiling = Math.max(ringStep * 2, Math.ceil((maxSpeed * 1.1) / ringStep) * ringStep);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = 'rgba(23,49,63,0.12)';
  ctx.lineWidth = 1;
  for (let ringIdx = 1; ringIdx <= 4; ringIdx++) {
    const ringRadius = radius * ringIdx / 4;
    ctx.beginPath();
    ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
  }

  const guideAngles = [0, 45, 90, 135, 180];
  ctx.strokeStyle = 'rgba(23,49,63,0.08)';
  for (const angle of guideAngles) {
    const starboard = polarToCanvasPoint(angle, speedCeiling, speedCeiling, 1, centerX, centerY, radius);
    const port = polarToCanvasPoint(angle, speedCeiling, speedCeiling, -1, centerX, centerY, radius);
    ctx.beginPath();
    if (angle === 0 || angle === 180) {
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(starboard.x, starboard.y);
    } else {
      ctx.moveTo(port.x, port.y);
      ctx.lineTo(starboard.x, starboard.y);
    }
    ctx.stroke();
  }

  ctx.fillStyle = 'rgba(23,49,63,0.78)';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let ringIdx = 1; ringIdx <= 4; ringIdx++) {
    const ringRadius = radius * ringIdx / 4;
    const ringValue = speedCeiling * ringIdx / 4;
    const label = ringValue < 4 ? ringValue.toFixed(1) : ringValue.toFixed(0);
    ctx.fillText(`${label} kt`, centerX + 8, centerY - ringRadius);
  }

  ctx.textAlign = 'center';
  ctx.fillText('0 deg TWA', centerX, centerY - radius - 20);
  ctx.fillText('180 deg TWA', centerX, centerY + radius + 22);
  for (const angle of [45, 90, 135]) {
    const right = polarToCanvasPoint(angle, speedCeiling * 1.04, speedCeiling, 1, centerX, centerY, radius);
    const left = polarToCanvasPoint(angle, speedCeiling * 1.04, speedCeiling, -1, centerX, centerY, radius);
    ctx.fillText(String(angle), right.x + 12, right.y);
    ctx.fillText(String(angle), left.x - 12, left.y);
  }

  const athleteCurve = buildPolarCurvePath(athleteRows, 'athleteSpeedKts', speedCeiling, centerX, centerY, radius);
  strokePolarCurve(ctx, athleteCurve, athleteStroke, 3.5);

  ctx.save();
  ctx.fillStyle = athleteStroke;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  for (const row of athleteRows) {
    const twaDeg = Number(row?.twaDeg);
    const speedKts = Number(row?.athleteSpeedKts);
    const starboard = polarToCanvasPoint(twaDeg, speedKts, speedCeiling, 1, centerX, centerY, radius);
    const port = polarToCanvasPoint(twaDeg, speedKts, speedCeiling, -1, centerX, centerY, radius);
    for (const point of [starboard, port]) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();

  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.font = '13px sans-serif';
  ctx.strokeStyle = athleteStroke;
  ctx.beginPath();
  ctx.moveTo(72, 60);
  ctx.lineTo(108, 60);
  ctx.stroke();
  ctx.fillStyle = athleteStroke;
  ctx.fillText('Athlete', 118, 60);

  return encodeCanvasPng(canvas);
}

function buildPolarCaptionText(polar, emptyMessage) {
  if (!polar || !hasPolarCurveData(polar)) return emptyMessage;
  const line1Parts = [];
  const statisticLabel = String(polar?.statisticLabel || '').trim();
  if (statisticLabel) line1Parts.push(statisticLabel);
  const binCount = countPolarCurveBins(polar);
  if (binCount > 0) line1Parts.push(`${binCount} bins`);

  const line2Parts = [];
  if (Number.isFinite(Number(polar?.wind?.directionDeg))) line2Parts.push(`from ${Math.round(Number(polar.wind.directionDeg))} deg`);
  if (Number.isFinite(Number(polar?.wind?.speedKts))) line2Parts.push(`${Number(polar.wind.speedKts).toFixed(1)} kt`);
  if (Number.isFinite(Number(polar?.wind?.inlierCount))) line2Parts.push(`${Math.round(Number(polar.wind.inlierCount))} inliers`);

  return [
    line1Parts.join(' | '),
    line2Parts.length ? `Wind ${line2Parts.join(' | ')}` : '',
  ].filter(Boolean).join('\n');
}

function drawPolarPlaceholder(pdf, x, y, w, h, message) {
  pdf.setDrawColor(221, 228, 233);
  pdf.setFillColor(249, 251, 252);
  pdf.roundedRect(x, y, w, h, 3, 3, 'FD');
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  pdf.setTextColor(120, 120, 120);
  const lines = pdf.splitTextToSize(String(message || ''), Math.max(10, w - 10));
  const lineHeight = 3.8;
  const startY = y + (h / 2) - ((Math.max(lines.length, 1) - 1) * lineHeight / 2);
  for (let i = 0; i < lines.length; i++) {
    pdf.text(lines[i], x + w / 2, startY + i * lineHeight, { align: 'center' });
  }
}

function drawSegmentPolarPages(pdf, segGroup, athleteColorMap, margin, segName, duration, localLabel, localRange) {
  if (!Array.isArray(segGroup) || segGroup.length === 0) return;

  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const epw = pageW - 2 * margin;
  const chartGap = 8;
  const colW = (epw - chartGap) / 2;
  const chartH = 78;
  const rowHeight = 110;
  const introText = 'Whole-session and analyzed-segment athlete polars from the fitted true wind. The plot keeps only real populated angle bins.';
  const athletes = [...segGroup].sort((a, b) => String(a?.athlete_name || '').localeCompare(String(b?.athlete_name || '')));

  const startPage = () => {
    pdf.addPage();
    let y = margin;
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(14);
    pdf.setTextColor(34, 34, 34);
    pdf.text(`Segment: ${segName} - Polar Plots`, margin + epw / 2, y + 6, { align: 'center' });
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(9);
    pdf.setTextColor(80, 80, 80);
    pdf.text(`Duration: ${Math.round(duration)}s`, margin + epw / 2, y + 11, { align: 'center' });
    if (localRange) {
      pdf.text(`${localLabel}: ${localRange}`, margin + epw / 2, y + 15, { align: 'center' });
      y += 18;
    } else {
      y += 14;
    }

    pdf.setFontSize(8);
    pdf.setTextColor(102, 102, 102);
    const introLines = pdf.splitTextToSize(introText, epw - 12);
    for (let i = 0; i < introLines.length; i++) {
      pdf.text(introLines[i], margin + epw / 2, y + 4 + i * 3.8, { align: 'center' });
    }
    return y + 6 + introLines.length * 3.8;
  };

  let y = startPage();
  for (const seg of athletes) {
    if (y + rowHeight > pageH - margin) y = startPage();

    const athleteName = seg?.athlete_name || 'Athlete';
    const athleteHex = athleteColorMap[athleteName] || athleteColorMap[seg?.athlete_id] || athleteColor(0);
    const [cr, cg, cb] = hexToRgb(validAthleteColor(athleteHex) ? athleteHex : athleteColor(0));
    const leftX = margin;
    const rightX = margin + colW + chartGap;

    pdf.setFillColor(cr, cg, cb);
    pdf.circle(margin + 2.2, y + 2.8, 1.3, 'F');
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(11);
    pdf.setTextColor(34, 34, 34);
    pdf.text(athleteName, margin + 6, y + 3.8);
    y += 8;

    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(8.5);
    pdf.setTextColor(78, 78, 78);
    pdf.text('Whole Session', leftX + colW / 2, y, { align: 'center' });
    pdf.text('Analyzed Segment', rightX + colW / 2, y, { align: 'center' });

    const chartY = y + 3;
    const sessionEmpty = 'Not enough matched CSV samples across the session.';
    const segmentEmpty = 'Not enough clean CSV samples inside this segment.';
    const sessionChart = renderPolarPlot(seg?.polar_session, athleteHex);
    const segmentChart = renderPolarPlot(seg?.polar_segment, athleteHex);

    if (sessionChart) addImageDataUrl(pdf, sessionChart, leftX, chartY, colW, chartH);
    else drawPolarPlaceholder(pdf, leftX, chartY, colW, chartH, sessionEmpty);

    if (segmentChart) addImageDataUrl(pdf, segmentChart, rightX, chartY, colW, chartH);
    else drawPolarPlaceholder(pdf, rightX, chartY, colW, chartH, segmentEmpty);

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(7.5);
    pdf.setTextColor(98, 98, 98);
    const captionY = chartY + chartH + 4;
    const leftLines = pdf.splitTextToSize(buildPolarCaptionText(seg?.polar_session, sessionEmpty), Math.max(20, colW - 4));
    const rightLines = pdf.splitTextToSize(buildPolarCaptionText(seg?.polar_segment, segmentEmpty), Math.max(20, colW - 4));
    const maxCaptionLines = Math.max(leftLines.length, rightLines.length, 1);
    for (let i = 0; i < leftLines.length; i++) {
      pdf.text(leftLines[i], leftX + colW / 2, captionY + i * 3.7, { align: 'center' });
    }
    for (let i = 0; i < rightLines.length; i++) {
      pdf.text(rightLines[i], rightX + colW / 2, captionY + i * 3.7, { align: 'center' });
    }

    y = captionY + maxCaptionLines * 3.7 + 8;
  }
}

function toMercator(lat, lon) {
  const x = lon * 20037508.34 / 180;
  const latRad = lat * Math.PI / 180;
  const y = Math.log(Math.tan(Math.PI / 4 + latRad / 2)) * 20037508.34 / Math.PI;
  return [x, y];
}

function fromMercator(mx, my) {
  const lon = mx * 180 / 20037508.34;
  const lat = (2 * Math.atan(Math.exp(my * Math.PI / 20037508.34)) - Math.PI / 2) * 180 / Math.PI;
  return [lat, lon];
}

// Safe min/max for large arrays (avoids stack overflow from spread)
function arrMin(arr) { let m = Infinity;  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]; return m; }
function arrMax(arr) { let m = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]; return m; }

// ── CartoDB Positron tile fetching (matches Python contextily) ────────

const TILE_SIZE = 256;
const CARTODB_POSITRON = 'https://a.basemaps.cartocdn.com/light_all';

function lon2tileX(lon, z) { return ((lon + 180) / 360) * (1 << z); }
function lat2tileY(lat, z) {
  const r = lat * Math.PI / 180;
  return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * (1 << z);
}

/**
 * Fetch CartoDB Positron tiles and draw them onto `ctx`.
 * Returns the proj function that maps Mercator [x,y] → canvas [px,py].
 */
async function drawTileBackground(ctx, mercBounds, canvasW, canvasH, pad) {
  const { xMin, xMax, yMin, yMax } = mercBounds;
  const [latMin, lonMin] = fromMercator(xMin, yMin);
  const [latMax, lonMax] = fromMercator(xMax, yMax);
  const latPad = (latMax - latMin) * 0.12;
  const lonPad = (lonMax - lonMin) * 0.12;
  const bLatMin = latMin - latPad;
  const bLatMax = latMax + latPad;
  const bLonMin = lonMin - lonPad;
  const bLonMax = lonMax + lonPad;

  // Choose zoom so we fetch ≤ 25 tiles
  let zoom = 16;
  for (let z = 18; z >= 1; z--) {
    const txMin = Math.floor(lon2tileX(bLonMin, z));
    const txMax = Math.floor(lon2tileX(bLonMax, z));
    const tyMin = Math.floor(lat2tileY(bLatMax, z));
    const tyMax = Math.floor(lat2tileY(bLatMin, z));
    if ((txMax - txMin + 1) * (tyMax - tyMin + 1) <= 25) { zoom = z; break; }
  }
  const txMin = Math.floor(lon2tileX(bLonMin, zoom));
  const txMax = Math.floor(lon2tileX(bLonMax, zoom));
  const tyMin = Math.floor(lat2tileY(bLatMax, zoom));
  const tyMax = Math.floor(lat2tileY(bLatMin, zoom));

  // Compute Mercator bounds of the data with padding
  const [mercPadXMin] = toMercator(bLatMin, bLonMin);
  const [mercPadXMax, mercPadYMax] = toMercator(bLatMax, bLonMax);
  const [, mercPadYMin] = toMercator(bLatMin, bLonMin);

  // Projection: Mercator → canvas (using padded data bounds → fill entire canvas)
  const mxRange = (mercPadXMax - mercPadXMin) || 1;
  const myRange = (mercPadYMax - mercPadYMin) || 1;
  const drawW = canvasW - 2 * pad;
  const drawH = canvasH - 2 * pad;
  const scale = Math.min(drawW / mxRange, drawH / myRange);
  const cx = canvasW / 2 - ((mercPadXMin + mercPadXMax) / 2) * scale;
  const cy = canvasH / 2 + ((mercPadYMin + mercPadYMax) / 2) * scale;
  const proj = ([mx, my]) => [cx + mx * scale, cy - my * scale];

  // Helper to get Mercator bounds of a tile
  function tileMercBounds(tx, ty, z) {
    const n = 1 << z;
    const lnMin = tx / n * 360 - 180;
    const lnMax = (tx + 1) / n * 360 - 180;
    const ltMax = Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n))) * 180 / Math.PI;
    const ltMin = Math.atan(Math.sinh(Math.PI * (1 - 2 * (ty + 1) / n))) * 180 / Math.PI;
    const [mxA, myA] = toMercator(ltMin, lnMin);
    const [mxB, myB] = toMercator(ltMax, lnMax);
    return { mxMin: mxA, myMin: myA, mxMax: mxB, myMax: myB };
  }

  // Fetch all tiles in parallel
  const tilePromises = [];
  for (let ty = tyMin; ty <= tyMax; ty++) {
    for (let tx = txMin; tx <= txMax; tx++) {
      const url = `${CARTODB_POSITRON}/${zoom}/${tx}/${ty}@2x.png`;
      tilePromises.push(
        loadImage(url).then(img => ({ img, tx, ty })).catch(() => null)
      );
    }
  }
  const tiles = (await Promise.all(tilePromises)).filter(Boolean);

  // Draw tiles as background
  ctx.save();
  for (const { img, tx, ty } of tiles) {
    const tb = tileMercBounds(tx, ty, zoom);
    const [cx0, cy0] = proj([tb.mxMin, tb.myMax]); // top-left (north-west)
    const [cx1, cy1] = proj([tb.mxMax, tb.myMin]); // bottom-right (south-east)
    ctx.drawImage(img, cx0, cy0, cx1 - cx0, cy1 - cy0);
  }
  ctx.restore();

  return proj;
}

// ── GPS map rendering (matches Python _make_map_image) ────────────────

async function renderGpsMap(gpsPaths, width = 1560, height = 840) {
  if (!gpsPaths.length || gpsPaths.every(p => !p.points?.length)) return null;

  const { canvas, ctx } = createCanvas(width, height);
  const PAD = 60;

  let allMerc = [];
  for (const path of gpsPaths) {
    if (!path.points?.length) continue;
    for (const p of path.points) allMerc.push(toMercator(p.lat, p.lon));
  }
  if (allMerc.length === 0) return null;

  const xMin = arrMin(allMerc.map(p => p[0]));
  const xMax = arrMax(allMerc.map(p => p[0]));
  const yMin = arrMin(allMerc.map(p => p[1]));
  const yMax = arrMax(allMerc.map(p => p[1]));

  // Fallback grey background (tiles will paint over this)
  ctx.fillStyle = '#f0f0f0';
  ctx.fillRect(0, 0, width, height);

  // Fetch & draw CartoDB tiles, get back projection aligned to tiles
  let proj;
  try {
    proj = await drawTileBackground(ctx, { xMin, xMax, yMin, yMax }, width, height, PAD);
  } catch (e) {
    console.warn('[report-pdf] tile fetch failed, using plain bg:', e);
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    const scale = Math.min((width - 2 * PAD) / xRange, (height - 2 * PAD) / yRange);
    const cx = width / 2 - (xMin + xMax) / 2 * scale;
    const cy = height / 2 + (yMin + yMax) / 2 * scale;
    proj = ([x, y]) => [cx + x * scale, cy - y * scale];
  }

  // Draw tracks (Python: linewidth=1.8, alpha=0.85)
  for (const path of gpsPaths) {
    if (!path.points?.length) continue;
    const merc = path.points.map(p => toMercator(p.lat, p.lon));
    const color = path.color || PALETTE[0];

    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.globalAlpha = 0.85;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    const [fx, fy] = proj(merc[0]);
    ctx.moveTo(fx, fy);
    for (let i = 1; i < merc.length; i++) {
      const [px, py] = proj(merc[i]);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Single direction arrow at midpoint, enlarged for PDF readability.
    const nPts = merc.length;
    if (nPts >= 4) {
      const mid = Math.floor(nPts / 2);
      const prev = Math.max(0, mid - 2);
      const [ax, ay] = proj(merc[prev]);
      const [bx, by] = proj(merc[mid]);
      const angle = Math.atan2(by - ay, bx - ax);
      const sz = 34;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.2;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(bx + sz * 0.4 * Math.cos(angle), by + sz * 0.4 * Math.sin(angle));
      ctx.lineTo(bx - sz * Math.cos(angle - 0.45), by - sz * Math.sin(angle - 0.45));
      ctx.lineTo(bx - sz * 0.3 * Math.cos(angle), by - sz * 0.3 * Math.sin(angle));
      ctx.lineTo(bx - sz * Math.cos(angle + 0.45), by - sz * Math.sin(angle + 0.45));
      ctx.closePath();
      ctx.stroke();
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
  }

  // Legend (Python: fontsize=7, upper left, framealpha=0.8) — always show for athlete identification
  {
    const legendEntries = gpsPaths.filter(p => p.label);
    if (legendEntries.length >= 1) {
      const legendH = legendEntries.length * 32 + 16;
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(8, 8, 240, legendH);
      ctx.globalAlpha = 1.0;
      ctx.font = '22px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      let ly = 32;
      for (const path of legendEntries) {
        ctx.fillStyle = path.color || PALETTE[0];
        ctx.fillRect(16, ly - 10, 22, 22);
        ctx.fillStyle = '#222';
        ctx.fillText(path.label, 46, ly + 2);
        ly += 32;
      }
    }
  }

  return encodeCanvasJpeg(canvas, 0.86);
}

// ── Summary map (matches Python _make_summary_map_image) ──────────────

async function renderSummaryMap(fullTrackPointsByFile, segments, athleteColorMap, width = 1680, height = 1080) {
  const fullPaths = [];
  for (const pts of Object.values(fullTrackPointsByFile || {})) {
    if (!Array.isArray(pts) || pts.length < 2) continue;
    const clean = [];
    for (const p of pts) {
      const lat = Number(p?.lat);
      const lon = Number(p?.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) clean.push({ lat, lon });
    }
    if (clean.length >= 2) fullPaths.push(clean);
  }

  const overlays = [];
  for (const seg of (segments || [])) {
    const pts = [];
    for (const p of (seg.gps_path || [])) {
      const lat = Number(p?.lat);
      const lon = Number(p?.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) pts.push({ lat, lon });
    }
    if (pts.length < 2) continue;
    overlays.push({
      points: pts,
      segName: String(seg.name || ''),
      athleteName: String(seg.athlete_name || 'Athlete'),
      athleteId: seg.athlete_id,
      color: athleteColorMap[seg.athlete_name] || athleteColorMap[seg.athlete_id] || PALETTE[0],
    });
  }

  // Collect all Mercator points (use push in loop, not spread — avoid stack overflow)
  const allMercX = [];
  const allMercY = [];
  for (const pts of fullPaths) {
    for (const p of pts) { const [mx, my] = toMercator(p.lat, p.lon); allMercX.push(mx); allMercY.push(my); }
  }
  for (const ov of overlays) {
    for (const p of ov.points) { const [mx, my] = toMercator(p.lat, p.lon); allMercX.push(mx); allMercY.push(my); }
  }
  if (allMercX.length < 2) return null;

  const PAD = 50;
  const { canvas, ctx } = createCanvas(width, height);
  const xMin = arrMin(allMercX);
  const xMax = arrMax(allMercX);
  const yMin = arrMin(allMercY);
  const yMax = arrMax(allMercY);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Fetch CartoDB tiles as background
  let proj;
  try {
    proj = await drawTileBackground(ctx, { xMin, xMax, yMin, yMax }, width, height, PAD);
  } catch (e) {
    console.warn('[report-pdf] summary map tile fetch failed:', e);
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    const scale = Math.min((width - 2 * PAD) / xRange, (height - 2 * PAD) / yRange);
    const cx = width / 2 - ((xMin + xMax) / 2) * scale;
    const cy = height / 2 + ((yMin + yMax) / 2) * scale;
    proj = ([x, y]) => [cx + x * scale, cy - y * scale];
  }

  // 1) Full session tracks in thin gray (Python: color="#bbbbbb", lw=0.8, alpha=0.6)
  ctx.save();
  ctx.strokeStyle = '#bbbbbb';
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 2.0;
  ctx.lineJoin = 'round';
  for (const pts of fullPaths) {
    if (pts.length < 2) continue;
    const merc = pts.map(p => toMercator(p.lat, p.lon));
    ctx.beginPath();
    const [x0, y0] = proj(merc[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < merc.length; i++) {
      const [x, y] = proj(merc[i]);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();

  // 2) Segment overlays (Python: color=athlete, lw=2.5, alpha=0.9, zorder=2)
  const legend = [];
  const legendSeen = new Set();
  for (const ov of overlays) {
    const merc = ov.points.map(p => toMercator(p.lat, p.lon));
    ctx.save();
    ctx.strokeStyle = ov.color;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    const [x0, y0] = proj(merc[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < merc.length; i++) {
      const [x, y] = proj(merc[i]);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Track athlete for legend (Python: one entry per athlete, not per segment)
    const legendKey = ov.athleteName;
    if (!legendSeen.has(legendKey)) {
      legendSeen.add(legendKey);
      legend.push({ label: ov.athleteName, color: ov.color });
    }

    // Segment label at midpoint, offset from track so line remains visible.
    const midIdx = Math.floor(merc.length / 2);
    const [mx, my] = proj(merc[midIdx]);
    const txt = ov.segName.slice(0, 18);
    if (txt) {
      const prevIdx = Math.max(0, midIdx - 2);
      const nextIdx = Math.min(merc.length - 1, midIdx + 2);
      const [px0, py0] = proj(merc[prevIdx]);
      const [px1, py1] = proj(merc[nextIdx]);
      let dx = px1 - px0;
      let dy = py1 - py0;
      const vlen = Math.hypot(dx, dy) || 1;
      dx /= vlen;
      dy /= vlen;
      const normalX = -dy;
      const normalY = dx;
      const labelOffset = 24;
      const lx = mx + normalX * labelOffset;
      const ly = my + normalY * labelOffset;

      ctx.font = 'bold 20px sans-serif';
      const tw = ctx.measureText(txt).width;
      const bw = tw + 16;
      const bh = 28;
      const bx = lx - bw / 2;
      const by = ly - bh / 2;
      ctx.save();
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = ov.color;
      _roundRect(ctx, bx, by, bw, bh, 6);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(lx, ly);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 20px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(txt, lx, ly + 1);
    }
  }

  // Legend (Python: Patch legend, fontsize=7, upper left, framealpha=0.8)
  if (legend.length > 0) {
    const legendH = legend.length * 32 + 16;
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(8, 8, 260, legendH);
    ctx.globalAlpha = 1.0;
    ctx.font = '22px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    let ly = 36;
    for (const item of legend) {
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = item.color;
      ctx.fillRect(16, ly - 18, 20, 20);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = '#222222';
      ctx.fillText(item.label, 44, ly);
      ly += 32;
    }
  }

  return encodeCanvasJpeg(canvas, 0.88);
}

// ── Hull STL loading ──────────────────────────────────────────────────

const HULL_OUTLINE_X = [-4.2, -3.5, -1.5, 0.0, 0.5, 0.0, -1.5, -3.5, -4.2];
const HULL_OUTLINE_Y = [0, 0.6, 0.73, 0.5, 0, -0.5, -0.73, -0.6, 0];
const HULL_SLICE_LEVELS = [-0.5, -1.0, -2.0, -5.0, -10.0];
// Keep STL→boat mapping consistent with live viewer / processing core.
const HULL_VIEWER_X_OFFSET_M = -2.974;

let _hullDetailCache = null;
let _hullDetailLoaded = false;

async function loadHullDetailFromStl() {
  if (_hullDetailLoaded) return _hullDetailCache || [];
  _hullDetailLoaded = true;
  try {
    let buffer = null;
    const candidates = [
      new URL('../Hull.stl', import.meta.url).href,
      '/api/assets/hull.stl',
      '/assets/hull.stl',
      '/static/Hull.stl',
    ];
    for (const url of candidates) {
      try {
        const resp = await fetch(url);
        if (!resp.ok) continue;
        buffer = await resp.arrayBuffer();
        if (buffer?.byteLength > 0) break;
      } catch { /* try next */ }
    }
    if (!buffer) return [];
    const triangles = parseStlTriangles(buffer);
    if (!triangles.length) return [];
    const sections = [];
    for (const depth of HULL_SLICE_LEVELS) {
      const segments = sliceTrianglesAtY(triangles, depth);
      if (segments.length) sections.push({ depth, segments });
    }
    _hullDetailCache = sections;
    return sections;
  } catch (err) {
    console.warn('[report-pdf] Failed to load/parse Hull.stl:', err);
    return [];
  }
}

function parseStlTriangles(buffer) {
  const dv = new DataView(buffer);
  if (buffer.byteLength >= 84) {
    const triCount = dv.getUint32(80, true);
    const expected = 84 + triCount * 50;
    if (expected === buffer.byteLength) {
      const out = [];
      let off = 84;
      for (let i = 0; i < triCount; i++) {
        off += 12; // skip normal
        const x0 = dv.getFloat32(off, true); off += 4;
        const y0 = dv.getFloat32(off, true); off += 4;
        const z0 = dv.getFloat32(off, true); off += 4;
        const x1 = dv.getFloat32(off, true); off += 4;
        const y1 = dv.getFloat32(off, true); off += 4;
        const z1 = dv.getFloat32(off, true); off += 4;
        const x2 = dv.getFloat32(off, true); off += 4;
        const y2 = dv.getFloat32(off, true); off += 4;
        const z2 = dv.getFloat32(off, true); off += 4;
        off += 2;
        out.push([x0, y0, z0, x1, y1, z1, x2, y2, z2]);
      }
      return out;
    }
  }
  // ASCII STL fallback
  const txt = new TextDecoder().decode(buffer);
  const nums = txt.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi);
  if (!nums || nums.length < 12) return [];
  const out = [];
  let i = 0;
  while (i + 11 < nums.length) {
    const x0 = Number(nums[i++]); const y0 = Number(nums[i++]); const z0 = Number(nums[i++]);
    const x1 = Number(nums[i++]); const y1 = Number(nums[i++]); const z1 = Number(nums[i++]);
    const x2 = Number(nums[i++]); const y2 = Number(nums[i++]); const z2 = Number(nums[i++]);
    if ([x0, y0, z0, x1, y1, z1, x2, y2, z2].every(Number.isFinite)) {
      out.push([x0, y0, z0, x1, y1, z1, x2, y2, z2]);
    }
  }
  return out;
}

function sliceTrianglesAtY(triangles, yLevel) {
  const EPS = 1e-6;
  const segments = [];
  function intersect(xa, ya, za, xb, yb, zb) {
    const da = ya - yLevel;
    const db = yb - yLevel;
    if (Math.abs(da) < EPS && Math.abs(db) < EPS) return null;
    if ((da > 0 && db > 0) || (da < 0 && db < 0)) return null;
    if (Math.abs(yb - ya) < EPS) return null;
    const t = (yLevel - ya) / (yb - ya);
    if (t < -EPS || t > 1 + EPS) return null;
    const x = xa + t * (xb - xa);
    const z = za + t * (zb - za);
    // Matches app.js + processing_core.py:
    //   boat_x = -stlZ/100 + offset, boat_y = -stlX/100
    return [(-z / 100.0) + HULL_VIEWER_X_OFFSET_M, -x / 100.0];
  }
  for (const tri of triangles) {
    const p = [];
    const i01 = intersect(tri[0], tri[1], tri[2], tri[3], tri[4], tri[5]);
    const i12 = intersect(tri[3], tri[4], tri[5], tri[6], tri[7], tri[8]);
    const i20 = intersect(tri[6], tri[7], tri[8], tri[0], tri[1], tri[2]);
    if (i01) p.push(i01);
    if (i12) p.push(i12);
    if (i20) p.push(i20);
    if (p.length < 2) continue;
    const uniq = [];
    for (const q of p) {
      const exists = uniq.some(u => Math.abs(u[0] - q[0]) < 1e-6 && Math.abs(u[1] - q[1]) < 1e-6);
      if (!exists) uniq.push(q);
    }
    if (uniq.length >= 2) segments.push([uniq[0], uniq[1]]);
  }
  return segments;
}

// Draw hull outline (matches Python _draw_hull_on_ax — depth-based styling)
function drawHullOutline(ctx, projX, projY, hullDetail = null, showCenterline = true) {
  if (Array.isArray(hullDetail) && hullDetail.length > 0) {
    for (const section of hullDetail) {
      const depth = Number(section?.depth);
      let stroke, alpha, width;
      if (depth >= -1) {
        stroke = '#444444'; alpha = 0.85; width = 1.4;
      } else if (depth >= -3) {
        stroke = '#777777'; alpha = 0.55; width = 0.9;
      } else {
        stroke = '#aaaaaa'; alpha = 0.35; width = 0.5;
      }
      ctx.save();
      ctx.strokeStyle = stroke;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      for (const seg of (section.segments || [])) {
        const [a, b] = seg;
        ctx.beginPath();
        ctx.moveTo(projX(a[0]), projY(a[1]));
        ctx.lineTo(projX(b[0]), projY(b[1]));
        ctx.stroke();
      }
      ctx.restore();
    }
    if (showCenterline) {
      // Centerline (Python: ax.axhline(0, ...))
      ctx.save();
      ctx.strokeStyle = '#cccccc';
      ctx.lineWidth = 0.4;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(projX(-5.0), projY(0));
      ctx.lineTo(projX(1.0), projY(0));
      ctx.stroke();
      ctx.restore();
    }
    return;
  }
  // Fallback simple boat shape (Python: bx/by arrays)
  ctx.save();
  ctx.strokeStyle = '#888888';
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  for (let i = 0; i < HULL_OUTLINE_X.length; i++) {
    const px = projX(HULL_OUTLINE_X[i]);
    const py = projY(HULL_OUTLINE_Y[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
}

function drawHullOutlineBowUp(ctx, projSideX, projForeY, hullDetail = null, showCenterline = true) {
  if (Array.isArray(hullDetail) && hullDetail.length > 0) {
    for (const section of hullDetail) {
      const depth = Number(section?.depth);
      let stroke, alpha, width;
      if (depth >= -1) {
        stroke = '#444444'; alpha = 0.85; width = 1.4;
      } else if (depth >= -3) {
        stroke = '#777777'; alpha = 0.55; width = 0.9;
      } else {
        stroke = '#aaaaaa'; alpha = 0.35; width = 0.5;
      }
      ctx.save();
      ctx.strokeStyle = stroke;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      for (const seg of (section.segments || [])) {
        const [a, b] = seg;
        // a/b are [fore, side] in boat frame
        ctx.beginPath();
        ctx.moveTo(projSideX(a[1]), projForeY(a[0]));
        ctx.lineTo(projSideX(b[1]), projForeY(b[0]));
        ctx.stroke();
      }
      ctx.restore();
    }
    if (showCenterline) {
      ctx.save();
      ctx.strokeStyle = '#cccccc';
      ctx.lineWidth = 0.4;
      ctx.globalAlpha = 0.4;
      const cx = projSideX(0);
      ctx.beginPath();
      ctx.moveTo(cx, projForeY(-5.0));
      ctx.lineTo(cx, projForeY(1.0));
      ctx.stroke();
      ctx.restore();
    }
    return;
  }

  // Fallback simple outline.
  ctx.save();
  ctx.strokeStyle = '#888888';
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  for (let i = 0; i < HULL_OUTLINE_X.length; i++) {
    const px = projSideX(HULL_OUTLINE_Y[i]);
    const py = projForeY(HULL_OUTLINE_X[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
}

// ── Heatmap rendering (matches Python _make_heatmap_image) ────────────
//
// Renders 2D KDE contourf-style heatmaps on the hull outline.
// Supports per_athlete (side-by-side) and combined overlay modes.
const PDF_HEATMAP_GRID_SIZE_X = 5.4; // Slightly wider X window to keep bow fully in-frame.

async function renderHeatmapPerAthlete(dataByAthlete, options = {}) {
  const {
    grid_size_x = PDF_HEATMAP_GRID_SIZE_X, grid_size_y = 3.0,
    grid_center_x = -1.0, grid_center_y = 0.0,
    resolution = 180,
    hullDetail = null,
    showMeanPos = false,
    title = '',
    layoutCols = 3,
    bowUp = true,
    includeOverlayPanel = false,
  } = options;

  const halfX = grid_size_x / 2;
  const halfY = grid_size_y / 2;
  const xLo = grid_center_x - halfX;
  const xHi = grid_center_x + halfX;
  const yLo = grid_center_y - halfY;
  const yHi = grid_center_y + halfY;
  const athletes = dataByAthlete
    .filter(d => (d.points?.length || 0) >= 3)
    .map(d => {
      const filteredPoints = (d.points || []).filter(([x, y]) => x >= xLo && x <= xHi && y >= yLo && y <= yHi);
      const canRenderDensity = filteredPoints.length >= 3;
      const meanX = canRenderDensity ? filteredPoints.reduce((sum, [x]) => sum + x, 0) / filteredPoints.length : null;
      const meanY = canRenderDensity ? filteredPoints.reduce((sum, [, y]) => sum + y, 0) / filteredPoints.length : null;
      return {
        ...d,
        filteredPoints,
        canRenderDensity,
        meanX,
        meanY,
        heatLayerCanvas: null,
      };
    });
  if (athletes.length === 0) return null;
  const hasOverlayPanel = includeOverlayPanel && athletes.length >= 2;
  const panelCount = athletes.length + (hasOverlayPanel ? 1 : 0);

  const panelW = bowUp ? 250 : 600;
  const panelH = bowUp
    ? Math.round(panelW * (grid_size_x / grid_size_y))
    : Math.round(panelW * (grid_size_y / grid_size_x));
  const GAP_X = 18;
  const GAP_Y = 18;
  const TITLE_H = title ? 24 : 0;
  const LABEL_H = 22;
  const LEGEND_H = 28;
  const nCols = Math.max(1, Math.min(layoutCols, panelCount));
  const nRows = Math.ceil(panelCount / nCols);
  const rowBlockH = LABEL_H + panelH;
  const totalW = nCols * panelW + (nCols - 1) * GAP_X + 20;
  const totalH = TITLE_H + nRows * rowBlockH + (nRows - 1) * GAP_Y + LEGEND_H + 18;

  const { canvas, ctx } = createCanvas(totalW, totalH);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, totalW, totalH);

  // Title (Python: fig.suptitle)
  if (title) {
    ctx.font = 'bold 13px sans-serif';
    ctx.fillStyle = '#222222';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(title, totalW / 2, 4);
  }

  const xGrid = Array.from({ length: resolution }, (_, i) => xLo + (xHi - xLo) * i / (resolution - 1));
  const yGrid = Array.from({ length: resolution }, (_, i) => yLo + (yHi - yLo) * i / (resolution - 1));
  const xHalfStep = xGrid.length > 1 ? (xGrid[1] - xGrid[0]) / 2 : (xHi - xLo) / 2;
  const yHalfStep = yGrid.length > 1 ? (yGrid[1] - yGrid[0]) / 2 : (yHi - yLo) / 2;
  const heatmapYield = createHeatmapYieldController();
  const levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
  const projStdXLocal = (x) => ((x - xLo) / (xHi - xLo)) * panelW;
  const projStdYLocal = (y) => ((yHi - y) / (yHi - yLo)) * panelH;
  // Keep bow-up side axis consistent with the in-app heatmap builder view.
  const projRotXLocal = (side) => ((yHi - side) / (yHi - yLo)) * panelW;
  const projRotYLocal = (fore) => ((xHi - fore) / (xHi - xLo)) * panelH;

  async function ensureAthleteHeatLayer(ath) {
    if (ath.heatLayerCanvas || !ath.canRenderDensity) return ath.heatLayerCanvas;
    const { canvas: layerCanvas, ctx: layerCtx } = createCanvas(panelW, panelH);
    const kde = await gaussianKde2dAsync(ath.filteredPoints, xGrid, yGrid, 0.45, heatmapYield);
    if (!kde) return null;

    const rgb = hexToRgb(ath.color || PALETTE[0]);
    for (let iy = 0; iy < kde.ny; iy++) {
      for (let ix = 0; ix < kde.nx; ix++) {
        const val = kde.Z[iy * kde.nx + ix];
        if (val < levels[0]) continue;

        let lvl = 0;
        for (const l of levels) { if (val >= l) lvl = l; }
        const t = lvl;
        const r = Math.round(255 + (rgb[0] - 255) * t);
        const g = Math.round(255 + (rgb[1] - 255) * t);
        const b = Math.round(255 + (rgb[2] - 255) * t);
        layerCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.92 * t})`;

        const x = xGrid[ix];
        const y = yGrid[iy];
        const x0 = Math.max(xLo, x - xHalfStep);
        const x1 = Math.min(xHi, x + xHalfStep);
        const y0 = Math.max(yLo, y - yHalfStep);
        const y1 = Math.min(yHi, y + yHalfStep);

        if (bowUp) {
          const px0 = projRotXLocal(y0);
          const px1 = projRotXLocal(y1);
          const pyTop = projRotYLocal(x1);
          const pyBottom = projRotYLocal(x0);
          layerCtx.fillRect(px0, pyTop, Math.max(0.5, px1 - px0), Math.max(0.5, pyBottom - pyTop));
        } else {
          const px0 = projStdXLocal(x0);
          const px1 = projStdXLocal(x1);
          const pyTop = projStdYLocal(y1);
          const pyBottom = projStdYLocal(y0);
          layerCtx.fillRect(px0, pyTop, Math.max(0.5, px1 - px0), Math.max(0.5, pyBottom - pyTop));
        }
      }
      await heatmapYield.maybeYield();
    }

    if (bowUp) {
      const rotNx = kde.ny;
      const rotNy = kde.nx;
      const rotZ = new Float64Array(rotNx * rotNy);
      for (let iy = 0; iy < kde.ny; iy++) {
        for (let ix = 0; ix < kde.nx; ix++) {
          rotZ[ix * rotNx + iy] = kde.Z[iy * kde.nx + ix];
        }
        await heatmapYield.maybeYield();
      }
      const rotKde = { Z: rotZ, nx: rotNx, ny: rotNy };
      await _drawContourLinesAsync(layerCtx, rotKde, yGrid, xGrid, [0.2, 0.5, 0.8], ath.color || PALETTE[0], 0.8, 0.5, projRotXLocal, projRotYLocal, heatmapYield);
    } else {
      await _drawContourLinesAsync(layerCtx, kde, xGrid, yGrid, [0.2, 0.5, 0.8], ath.color || PALETTE[0], 0.8, 0.5, projStdXLocal, projStdYLocal, heatmapYield);
    }

    ath.heatLayerCanvas = layerCanvas;
    await heatmapYield.maybeYield();
    return layerCanvas;
  }

  for (let pi = 0; pi < panelCount; pi++) {
    const isOverlayPanel = hasOverlayPanel && pi === panelCount - 1;
    const ath = isOverlayPanel ? null : athletes[pi];
    const row = Math.floor(pi / nCols);
    const col = pi % nCols;
    const ox = 10 + col * (panelW + GAP_X);
    const oy = TITLE_H + row * (rowBlockH + GAP_Y) + LABEL_H;

    // Panel label
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = isOverlayPanel ? '#444444' : (ath.color || '#444444');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(isOverlayPanel ? 'Overlay' : (ath.label || `Athlete ${pi + 1}`), ox + panelW / 2, oy + 10);

    const projStdX = (x) => ox + ((x - xLo) / (xHi - xLo)) * panelW;
    const projStdY = (y) => oy + ((yHi - y) / (yHi - yLo)) * panelH;
    const projRotX = (side) => ox + ((yHi - side) / (yHi - yLo)) * panelW;
    const projRotY = (fore) => oy + ((xHi - fore) / (xHi - xLo)) * panelH;

    // White background for panel
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(ox, oy, panelW, panelH);

    if (bowUp) {
      drawHullOutlineBowUp(ctx, projRotX, projRotY, hullDetail, false);
    } else {
      drawHullOutline(ctx, projStdX, projStdY, hullDetail, false);
    }

    if (isOverlayPanel) {
      for (const a of athletes) {
        const heatLayer = await ensureAthleteHeatLayer(a);
        if (!heatLayer) continue;
        ctx.save();
        ctx.globalAlpha = 0.78;
        ctx.drawImage(heatLayer, ox, oy);
        ctx.restore();
      }
    } else {
      const heatLayer = await ensureAthleteHeatLayer(ath);
      if (heatLayer) {
        ctx.drawImage(heatLayer, ox, oy);
      }

      if (showMeanPos && ath.canRenderDensity) {
        const lbl = `Fwd=${ath.meanX.toFixed(2)}m  Side=${ath.meanY.toFixed(2)}m`;
        ctx.font = 'bold 8px sans-serif';
        const tw = ctx.measureText(lbl).width;
        const lx = ox + 6;
        const ly = oy + panelH - 8;
        ctx.save();
        ctx.globalAlpha = 0.84;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = ath.color || PALETTE[0];
        ctx.lineWidth = 0.5;
        _roundRect(ctx, lx - 2, ly - 12, tw + 8, 14, 3);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = ath.color || PALETTE[0];
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(lbl, lx + 2, ly);
      }
    }

    if (bowUp) {
      drawHullOutlineBowUp(ctx, projRotX, projRotY, hullDetail, false);
    } else {
      drawHullOutline(ctx, projStdX, projStdY, hullDetail, false);
    }
    await heatmapYield.maybeYield();
  }

  // Legend row at bottom — always show for athlete name+color consistency
  {
    const legendY = TITLE_H + nRows * rowBlockH + (nRows - 1) * GAP_Y + 6;
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    // center the legend items
    const itemW = 100;
    const totalLegendW = athletes.length * itemW;
    let lx = (totalW - totalLegendW) / 2;
    for (const ath of athletes) {
      ctx.fillStyle = ath.color || PALETTE[0];
      ctx.fillRect(lx, legendY, 14, 14);
      ctx.fillStyle = '#222222';
      ctx.fillText(ath.label || '', lx + 18, legendY + 7);
      lx += itemW;
    }
  }

  await heatmapYield.maybeYield(true);
  return { dataUrl: encodeCanvasPng(canvas), width: totalW, height: totalH };
}

// Combined overlay heatmap (matches Python per_athlete=False path)
function renderHeatmapOverlay(dataByAthlete, options = {}) {
  const {
    width = 800, height = 480,
    grid_size_x = PDF_HEATMAP_GRID_SIZE_X, grid_size_y = 3.0,
    grid_center_x = -1.0, grid_center_y = 0.0,
    resolution = 200,
    hullDetail = null,
    showMeanPos = false,
    title = '',
  } = options;

  const athletes = dataByAthlete.filter(d => (d.points?.length || 0) >= 3);
  if (athletes.length === 0) return null;

  const { canvas, ctx } = createCanvas(width, height);
  const PAD = 30;
  const TITLE_H = title ? 18 : 0;

  const halfX = grid_size_x / 2;
  const halfY = grid_size_y / 2;
  const xLo = grid_center_x - halfX;
  const xHi = grid_center_x + halfX;
  const yLo = grid_center_y - halfY;
  const yHi = grid_center_y + halfY;

  // Keep hull orientation unchanged.
  const projX = (x) => PAD + ((x - xLo) / (xHi - xLo)) * (width - 2 * PAD);
  const projY = (y) => PAD + TITLE_H + ((yHi - y) / (yHi - yLo)) * (height - 2 * PAD - TITLE_H);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  if (title) {
    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#222222';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(title, width / 2, 4);
  }

  drawHullOutline(ctx, projX, projY, hullDetail, false);

  const xGrid = Array.from({ length: resolution }, (_, i) => xLo + (xHi - xLo) * i / (resolution - 1));
  const yGrid = Array.from({ length: resolution }, (_, i) => yLo + (yHi - yLo) * i / (resolution - 1));

  const plotAreaW = width - 2 * PAD;
  const plotAreaH = height - 2 * PAD - TITLE_H;

  for (const ath of athletes) {
    const pts = (ath.points || []).filter(([x, y]) => x >= xLo && x <= xHi && y >= yLo && y <= yHi);
    if (pts.length < 3) continue;

    const kde = gaussianKde2d(pts, xGrid, yGrid, 0.45);
    if (!kde) continue;

    const rgb = hexToRgb(ath.color || PALETTE[0]);
    const levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    const xHalfStep = xGrid.length > 1 ? (xGrid[1] - xGrid[0]) / 2 : (xHi - xLo) / 2;
    const yHalfStep = yGrid.length > 1 ? (yGrid[1] - yGrid[0]) / 2 : (yHi - yLo) / 2;

    for (let iy = 0; iy < kde.ny; iy++) {
      for (let ix = 0; ix < kde.nx; ix++) {
        const val = kde.Z[iy * kde.nx + ix];
        if (val < levels[0]) continue;
        let lvl = 0;
        for (const l of levels) { if (val >= l) lvl = l; }
        const t = lvl;
        const r = Math.round(255 + (rgb[0] - 255) * t);
        const g = Math.round(255 + (rgb[1] - 255) * t);
        const b = Math.round(255 + (rgb[2] - 255) * t);
        const alpha = 0.92 * t;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        const x = xGrid[ix];
        const y = yGrid[iy];
        const x0 = Math.max(xLo, x - xHalfStep);
        const x1 = Math.min(xHi, x + xHalfStep);
        const y0 = Math.max(yLo, y - yHalfStep);
        const y1 = Math.min(yHi, y + yHalfStep);
        const px0 = projX(x0);
        const px1 = projX(x1);
        const pyTop = projY(y1);
        const pyBottom = projY(y0);
        ctx.fillRect(px0, pyTop, Math.max(0.5, px1 - px0), Math.max(0.5, pyBottom - pyTop));
      }
    }

    _drawContourLines(ctx, kde, xGrid, yGrid, [0.2, 0.5, 0.8], ath.color || PALETTE[0], 0.8, 0.5, projX, projY);
  }

  drawHullOutline(ctx, projX, projY, hullDetail, false);

  // Mean position annotations (Python: per-athlete offset text boxes)
  if (showMeanPos) {
    let yOffset = 0.28;
    for (const ath of athletes) {
      const pts = (ath.points || []).filter(([x, y]) => x >= xLo && x <= xHi && y >= yLo && y <= yHi);
      if (pts.length < 3) continue;
      const meanXval = pts.reduce((a, p) => a + p[0], 0) / pts.length;
      const meanYval = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      const txt = `${ath.label}: Fwd=${meanXval.toFixed(2)}m  Side=${meanYval.toFixed(2)}m`;
      const lx = projX(xLo + 0.12);
      const ly = projY(yLo + yOffset);

      ctx.font = 'bold 8px sans-serif';
      const tw = ctx.measureText(txt).width;
      ctx.save();
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = ath.color || PALETTE[0];
      ctx.lineWidth = 0.5;
      _roundRect(ctx, lx - 2, ly - 12, tw + 8, 14, 3);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = ath.color || PALETTE[0];
      ctx.font = 'bold 8px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(txt, lx + 2, ly);
      yOffset += 0.35;
    }
  }

  // Legend (Python: upper right, Patch legend) — always show for athlete identification
  if (athletes.length >= 1) {
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let ly = PAD + TITLE_H + 8;
    for (const ath of athletes) {
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = ath.color || PALETTE[0];
      ctx.fillRect(width - PAD - 80, ly, 10, 10);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = '#222222';
      ctx.fillText(ath.label || '', width - PAD - 66, ly + 5);
      ly += 14;
    }
  }

  return canvas.toDataURL('image/png');
}

// Approximate contour lines by marching squares on a grid
async function _drawContourLinesAsync(ctx, kde, xGrid, yGrid, levels, color, lineWidth, alpha, projX, projY, yieldController = null) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;

  for (const level of levels) {
    for (let iy = 0; iy < kde.ny - 1; iy++) {
      for (let ix = 0; ix < kde.nx - 1; ix++) {
        const v00 = kde.Z[iy * kde.nx + ix];
        const v10 = kde.Z[iy * kde.nx + ix + 1];
        const v01 = kde.Z[(iy + 1) * kde.nx + ix];
        const v11 = kde.Z[(iy + 1) * kde.nx + ix + 1];

        const edges = [];
        if ((v00 - level) * (v10 - level) < 0) {
          const t = (level - v00) / (v10 - v00);
          edges.push([xGrid[ix] + t * (xGrid[ix + 1] - xGrid[ix]), yGrid[iy]]);
        }
        if ((v10 - level) * (v11 - level) < 0) {
          const t = (level - v10) / (v11 - v10);
          edges.push([xGrid[ix + 1], yGrid[iy] + t * (yGrid[iy + 1] - yGrid[iy])]);
        }
        if ((v01 - level) * (v11 - level) < 0) {
          const t = (level - v01) / (v11 - v01);
          edges.push([xGrid[ix] + t * (xGrid[ix + 1] - xGrid[ix]), yGrid[iy + 1]]);
        }
        if ((v00 - level) * (v01 - level) < 0) {
          const t = (level - v00) / (v01 - v00);
          edges.push([xGrid[ix], yGrid[iy] + t * (yGrid[iy + 1] - yGrid[iy])]);
        }
        if (edges.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(projX(edges[0][0]), projY(edges[0][1]));
          ctx.lineTo(projX(edges[1][0]), projY(edges[1][1]));
          ctx.stroke();
        }
      }
      if (yieldController) await yieldController.maybeYield();
    }
  }
  ctx.restore();
}

function _drawContourLines(ctx, kde, xGrid, yGrid, levels, color, lineWidth, alpha, projX, projY) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;

  for (const level of levels) {
    for (let iy = 0; iy < kde.ny - 1; iy++) {
      for (let ix = 0; ix < kde.nx - 1; ix++) {
        const v00 = kde.Z[iy * kde.nx + ix];
        const v10 = kde.Z[iy * kde.nx + ix + 1];
        const v01 = kde.Z[(iy + 1) * kde.nx + ix];
        const v11 = kde.Z[(iy + 1) * kde.nx + ix + 1];

        // Marching squares: find edges that cross the level
        const edges = [];
        if ((v00 - level) * (v10 - level) < 0) {
          const t = (level - v00) / (v10 - v00);
          edges.push([xGrid[ix] + t * (xGrid[ix + 1] - xGrid[ix]), yGrid[iy]]);
        }
        if ((v10 - level) * (v11 - level) < 0) {
          const t = (level - v10) / (v11 - v10);
          edges.push([xGrid[ix + 1], yGrid[iy] + t * (yGrid[iy + 1] - yGrid[iy])]);
        }
        if ((v01 - level) * (v11 - level) < 0) {
          const t = (level - v01) / (v11 - v01);
          edges.push([xGrid[ix] + t * (xGrid[ix + 1] - xGrid[ix]), yGrid[iy + 1]]);
        }
        if ((v00 - level) * (v01 - level) < 0) {
          const t = (level - v00) / (v01 - v00);
          edges.push([xGrid[ix], yGrid[iy] + t * (yGrid[iy + 1] - yGrid[iy])]);
        }
        if (edges.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(projX(edges[0][0]), projY(edges[0][1]));
          ctx.lineTo(projX(edges[1][0]), projY(edges[1][1]));
          ctx.stroke();
        }
      }
    }
  }
  ctx.restore();
}

// ── Scatter plot (matches Python style) ───────────────────────────────

function renderScatter(dataByAthlete, xlabel, ylabel, fitLine = null) {
  const W = 660, H = 400;
  const PAD = { top: 20, right: 20, bottom: 40, left: 55 };
  const { canvas, ctx } = createCanvas(W, H);
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const allX = [], allY = [];
  for (const d of dataByAthlete) {
    for (const [x, y] of (d.points || [])) { allX.push(x); allY.push(y); }
  }
  if (allX.length === 0) return null;

  const xMin = Math.min(...allX) * 0.9;
  const xMax = Math.max(...allX) * 1.1 || 1;
  const yMin = Math.min(...allY) * 0.9;
  const yMax = Math.max(...allY) * 1.1 || 1;

  // Grid
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
  }

  // Points
  for (const d of dataByAthlete) {
    const rgb = hexToRgb(d.color || PALETTE[0]);
    ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.4)`;
    for (const [x, y] of (d.points || [])) {
      const px = PAD.left + ((x - xMin) / (xMax - xMin)) * pw;
      const py = PAD.top + ph - ((y - yMin) / (yMax - yMin)) * ph;
      ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Fit line
  if (fitLine && fitLine.slope != null) {
    const x0 = xMin, x1 = xMax;
    const y0f = fitLine.slope * x0 + fitLine.intercept;
    const y1f = fitLine.slope * x1 + fitLine.intercept;
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(PAD.left + ((x0 - xMin) / (xMax - xMin)) * pw,
               PAD.top + ph - ((y0f - yMin) / (yMax - yMin)) * ph);
    ctx.lineTo(PAD.left + ((x1 - xMin) / (xMax - xMin)) * pw,
               PAD.top + ph - ((y1f - yMin) / (yMax - yMin)) * ph);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Axis labels
  ctx.fillStyle = '#222';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(xlabel, PAD.left + pw / 2, H - 2);
  ctx.save();
  ctx.translate(14, PAD.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();

  // Tick labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 5; i++) {
    const val = xMin + (xMax - xMin) * i / 5;
    ctx.fillText(val.toFixed(1), PAD.left + (pw * i) / 5, PAD.top + ph + 5);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const val = yMin + (yMax - yMin) * (4 - i) / 4;
    ctx.fillText(val.toFixed(1), PAD.left - 5, PAD.top + (ph * i) / 4);
  }

  return canvas.toDataURL('image/png');
}

// ── Timeline chart ────────────────────────────────────────────────────

function renderTimeline(seriesByAthlete, ylabel, title = '') {
  const W = 660, H = 220;
  const PAD = { top: 25, right: 20, bottom: 35, left: 55 };
  const { canvas, ctx } = createCanvas(W, H);
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  let tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (const { data } of seriesByAthlete) {
    for (const { t, v } of data) {
      if (t < tMin) tMin = t; if (t > tMax) tMax = t;
      if (v < vMin) vMin = v; if (v > vMax) vMax = v;
    }
  }
  if (!isFinite(tMin)) return null;
  const tRange = (tMax - tMin) || 1;
  const vRange = (vMax - vMin) || 1;

  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
  }

  for (const { data, color } of seriesByAthlete) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = PAD.left + ((data[i].t - tMin) / tRange) * pw;
      const y = PAD.top + ph - ((data[i].v - vMin) / vRange) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  if (title) {
    ctx.fillStyle = '#222';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, W / 2, 14);
  }

  ctx.fillStyle = '#222';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const val = vMin + vRange * (4 - i) / 4;
    ctx.fillText(val.toFixed(1), PAD.left - 4, PAD.top + (ph * i) / 4);
  }

  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(ylabel, PAD.left + pw / 2, H - 2);

  return canvas.toDataURL('image/png');
}

// ── Stats formatting (matches Python _format_stat / _format_stat_pm) ──

function fmtStat(val, dec = 1) {
  if (val == null) return '-';
  return val.toFixed(dec);
}

function fmtStatPm(avg, std, dec = 1) {
  if (avg == null) return '-';
  if (std == null || std <= 0) return avg.toFixed(dec);
  return `${avg.toFixed(dec)} \u00B1 ${std.toFixed(dec)}`;
}

function localTimeZone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch {
    return 'UTC';
  }
}

function isEpochSeconds(v) {
  return Number.isFinite(Number(v)) && Number(v) > 946684800; // >= 2000-01-01
}

function formatDateTimeInZone(epochSec, timeZone = null) {
  if (!isEpochSeconds(epochSec)) return '-';
  const d = new Date(Number(epochSec) * 1000);
  const tz = isValidTimeZone(timeZone) ? timeZone : null;
  const optsBase = tz ? { timeZone: tz } : {};
  const date = d.toLocaleDateString(undefined, {
    ...optsBase,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  const time = d.toLocaleTimeString(undefined, {
    ...optsBase,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    hourCycle: 'h23',
  });
  return `${date} ${time}`;
}

function formatLocalDateTime(epochSec) {
  return formatDateTimeInZone(epochSec, null);
}

function isValidTimeZone(tz) {
  if (!tz || typeof tz !== 'string') return false;
  try {
    Intl.DateTimeFormat(undefined, { timeZone: tz }).format(new Date());
    return true;
  } catch {
    return false;
  }
}

function formatSegmentRangeInZone(startEpochSec, endEpochSec, timeZone = null, locationLabel = null) {
  if (!isEpochSeconds(startEpochSec) || !isEpochSeconds(endEpochSec)) return null;
  const start = new Date(Number(startEpochSec) * 1000);
  const end = new Date(Number(endEpochSec) * 1000);
  const tz = isValidTimeZone(timeZone) ? timeZone : null;
  const optsBase = tz ? { timeZone: tz } : {};
  const sameDate =
    start.toLocaleDateString(undefined, { ...optsBase, year: 'numeric', month: '2-digit', day: '2-digit' }) ===
    end.toLocaleDateString(undefined, { ...optsBase, year: 'numeric', month: '2-digit', day: '2-digit' });
  const startDate = start.toLocaleDateString(undefined, { ...optsBase, year: 'numeric', month: '2-digit', day: '2-digit' });
  const startTime = start.toLocaleTimeString(undefined, {
    ...optsBase,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    hourCycle: 'h23',
  });
  const endDate = end.toLocaleDateString(undefined, { ...optsBase, year: 'numeric', month: '2-digit', day: '2-digit' });
  const endTime = end.toLocaleTimeString(undefined, {
    ...optsBase,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    hourCycle: 'h23',
  });
  const label = typeof locationLabel === 'string' ? locationLabel.trim() : '';
  const suffix = label ? ` (${label})` : '';
  if (sameDate) return `${startDate} ${startTime} - ${endTime}${suffix}`;
  return `${startDate} ${startTime} - ${endDate} ${endTime}${suffix}`;
}

function formatLocalSegmentRange(startEpochSec, endEpochSec) {
  return formatSegmentRangeInZone(startEpochSec, endEpochSec, null);
}

function _timezoneByLongitude(lon) {
  if (!Number.isFinite(Number(lon))) return null;
  // Approximate civil timezone from longitude (fallback when online lookup fails).
  const offset = Math.max(-12, Math.min(14, Math.round(Number(lon) / 15)));
  if (offset === 0) return 'Etc/GMT';
  // Etc/GMT signs are inverted: Etc/GMT-2 = UTC+2
  return `Etc/GMT${offset > 0 ? '-' : '+'}${Math.abs(offset)}`;
}

function _collectSegmentGpsPoints(rows, maxPoints = 1200) {
  const out = [];
  for (const row of rows || []) {
    for (const p of row?.gps_path || []) {
      const lat = Number(p?.lat);
      const lon = Number(p?.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) out.push([lat, lon]);
    }
  }
  if (out.length <= maxPoints) return out;
  const step = (out.length - 1) / (maxPoints - 1);
  const sampled = [];
  for (let i = 0; i < maxPoints; i++) sampled.push(out[Math.round(i * step)]);
  return sampled;
}

function _centroidLatLon(points) {
  if (!points?.length) return null;
  let latSum = 0;
  let lonSum = 0;
  for (const [lat, lon] of points) { latSum += lat; lonSum += lon; }
  return { lat: latSum / points.length, lon: lonSum / points.length };
}

const _tzLookupCache = new Map();

async function _fetchJsonWithTimeout(url, timeoutMs = 2200) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      signal: ctrl.signal,
      headers: { Accept: 'application/json' },
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } finally {
    clearTimeout(timer);
  }
}

function _labelFromTimeZone(timeZone) {
  if (!isValidTimeZone(timeZone)) return null;
  const tz = String(timeZone).trim();
  if (!tz || tz.startsWith('Etc/')) return null;
  const part = tz.includes('/') ? tz.split('/').pop() : tz;
  if (!part) return null;
  return part.replace(/_/g, ' ');
}

function _cleanPlaceLabel(label) {
  if (typeof label !== 'string') return null;
  const clean = label.trim();
  if (!clean || clean.includes('/')) return null;
  return clean;
}

function _extractPlaceLabelFromReverseGeocode(data) {
  if (!data || typeof data !== 'object') return null;

  // Prefer island-style informative labels when present (e.g. Lanzarote).
  const informative = Array.isArray(data?.localityInfo?.informative) ? data.localityInfo.informative : [];
  for (const item of informative) {
    const name = _cleanPlaceLabel(item?.name);
    const desc = String(item?.description || '').toLowerCase();
    if (!name) continue;
    if (desc.includes('island')) return name;
  }

  const directCandidates = [
    data?.locality,
    data?.city,
    data?.town,
    data?.village,
    data?.county,
    data?.principalSubdivision,
  ];
  for (const candidate of directCandidates) {
    const clean = _cleanPlaceLabel(candidate);
    if (clean) return clean;
  }

  // Fall back to informative names while skipping timezone/ocean/continent labels.
  for (const item of informative) {
    const name = _cleanPlaceLabel(item?.name);
    const desc = String(item?.description || '').toLowerCase();
    if (!name) continue;
    if (desc.includes('time zone') || desc.includes('ocean') || desc.includes('continent')) continue;
    return name;
  }

  return null;
}

async function _lookupPlaceLabel(lat, lon) {
  try {
    const url = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&localityLanguage=en`;
    const data = await _fetchJsonWithTimeout(url, 2600);
    return _extractPlaceLabelFromReverseGeocode(data);
  } catch {
    return null;
  }
}

async function resolveGpsTimeZone(rows, fallbackTz = null) {
  const localTz = localTimeZone();
  const fallback = isValidTimeZone(fallbackTz)
    ? fallbackTz
    : (isValidTimeZone(localTz) ? localTz : 'UTC');
  const pts = _collectSegmentGpsPoints(rows);
  const centroid = _centroidLatLon(pts);
  if (!centroid) return { timeZone: fallback, source: 'fallback' };

  const lat = Math.max(-90, Math.min(90, centroid.lat));
  const lon = Math.max(-180, Math.min(180, centroid.lon));
  const cacheKey = `${lat.toFixed(1)},${lon.toFixed(1)}`;
  if (_tzLookupCache.has(cacheKey)) return _tzLookupCache.get(cacheKey);

  let locationLabel = await _lookupPlaceLabel(lat, lon);

  // 1) Best-effort online lookup (IANA timezone from coordinates)
  try {
    const url = `https://timeapi.io/api/TimeZone/coordinate?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}`;
    const data = await _fetchJsonWithTimeout(url);
    const tz = String(
      data?.timeZone || data?.timezone || data?.ianaTimeZone || data?.zoneName || ''
    ).trim();
    if (isValidTimeZone(tz)) {
      if (!locationLabel) locationLabel = _labelFromTimeZone(tz);
      const info = { timeZone: tz, source: 'gps', locationLabel };
      _tzLookupCache.set(cacheKey, info);
      return info;
    }
  } catch {
    // fall through
  }

  // 2) Longitude-based approximation fallback (still GPS-derived)
  const approx = _timezoneByLongitude(lon);
  if (isValidTimeZone(approx)) {
    if (!locationLabel) locationLabel = _labelFromTimeZone(approx);
    const info = { timeZone: approx, source: 'approx', locationLabel };
    _tzLookupCache.set(cacheKey, info);
    return info;
  }

  if (!locationLabel) locationLabel = _labelFromTimeZone(fallback);
  const info = { timeZone: fallback, source: 'fallback', locationLabel };
  _tzLookupCache.set(cacheKey, info);
  return info;
}

// ── Metric definitions (identical to Python) ──────────────────────────

const METRIC_DEFS = [
  ['Avg SOG (kts)',          'sog',         'avg', 1, 'avg_sog'],
  ['Max SOG (kts)',          'sog',         'max', 1, 'max_sog'],
  ['Avg Heel (deg)',         'heel',        'avg', 1, 'avg_heel'],
  ['Max Heel (deg)',         'heel',        'max', 1, 'max_heel'],
  ['Avg Roll Moment (Nm)',   'moment_roll', 'avg', 0, 'avg_moment_roll'],
  ['Max Roll Moment (Nm)',   'moment_roll', 'max', 0, 'max_moment_roll'],
  ['Avg Trunk Angle (deg)',  'trunk_angle', 'avg', 1, 'avg_trunk_angle'],
  ['Max Trunk Angle (deg)',  'trunk_angle', 'max', 1, 'max_trunk_angle'],
  ['Avg Rudder Angle (deg)', 'rudder',      'avg', 1, null],
  ['Max Rudder Angle (deg)', 'rudder',      'max', 1, null],
  ['Avg Boom Angle (deg)',   'boom',        'avg', 1, null],
  ['Max Boom Angle (deg)',   'boom',        'max', 1, null],
];

function reportMetricDefs(opts = {}) {
  const includeBoomAngle = opts?.includeBoomAngle !== false;
  return METRIC_DEFS.filter(([, statKey]) => includeBoomAngle || statKey !== 'boom');
}

function isGold(golds, goldKey, splitId, athleteName) {
  if (!goldKey) return false;
  const g = golds?.[goldKey];
  if (!g || typeof g !== 'object') return false;
  return g[splitId] === athleteName;
}

function formatMetricCell(segData, metricDef) {
  const [, statKey, subKey, decimals] = metricDef;
  if (!segData) return '-';
  const stats = segData[statKey] || {};
  const val = stats[subKey];
  const std = stats.std;
  return subKey === 'max' ? fmtStat(val, decimals) : fmtStatPm(val, std, decimals);
}

// ── Summary table (matches Python _draw_summary_table) ────────────────
// Uses jsPDF manual drawing for the complex multi-row header with
// athlete-coloured sub-headers, coloured text, and gold dots.

function drawSummaryTable(pdf, segments, athletes, athleteColorMap, golds, margin, metricDefs = METRIC_DEFS) {
  const orderedSegments = sortSegmentsByRecordingTime(segments);
  const segGroups = groupSegmentsBySegId(orderedSegments);

  // Ordered unique split IDs + athlete names derived from SEGMENTS
  // (athletes[].name may differ from segments[].athlete_name)
  const splitIdsOrdered = [...segGroups.keys()];
  const athleteNamesOrdered = [];
  for (const seg of orderedSegments) {
    if (!athleteNamesOrdered.includes(seg.athlete_name)) athleteNamesOrdered.push(seg.athlete_name);
  }
  const nAth = athleteNamesOrdered.length;
  if (nAth === 0 || splitIdsOrdered.length === 0) return;

  const epw = pdf.internal.pageSize.getWidth() - 2 * margin;
  const metricColW = 36;
  const minDataColW = 14 * PDF_FONT_SCALE;
  const maxDataCols = Math.max(nAth, Math.floor((epw - metricColW) / minDataColW));
  const segmentsPerChunk = Math.max(1, Math.floor(maxDataCols / nAth));
  const splitChunks = [];
  for (let i = 0; i < splitIdsOrdered.length; i += segmentsPerChunk) {
    splitChunks.push(splitIdsOrdered.slice(i, i + segmentsPerChunk));
  }

  const rowH = 6.5 * PDF_FONT_SCALE;
  const headerH = 6.0 * PDF_FONT_SCALE;
  const pageBottom = pdf.internal.pageSize.getHeight() - 20;

  // Build segment lookup: (split_id, athlete_name) → segment data
  const segLookup = {};
  for (const s of segments) {
    segLookup[`${s.split_id}||${s.athlete_name}`] = s;
  }

  for (let chunkIdx = 0; chunkIdx < splitChunks.length; chunkIdx++) {
    const splitIds = splitChunks[chunkIdx];
    const nSeg = splitIds.length;
    const nDataCols = nSeg * nAth;
    let colW = Math.min((epw - metricColW) / Math.max(nDataCols, 1), 40);
    if (!Number.isFinite(colW) || colW <= 0) colW = minDataColW;
    colW = Math.max(colW, minDataColW);
    let totalW = metricColW + colW * nDataCols;
    if (totalW > epw) {
      colW = (epw - metricColW) / Math.max(nDataCols, 1);
      totalW = metricColW + colW * nDataCols;
    }
    const tableX = margin + (epw - totalW) / 2;

    const needed = headerH * 2 + rowH * metricDefs.length + (chunkIdx < splitChunks.length - 1 ? 4 : 0);
    if (pdf.getY() + needed > pageBottom) pdf.addPage();

    const y0 = pdf.getY();

    // Metric column header (spans two rows)
    pdf.setFillColor(...TABLE_METRIC_HEADER_BG);
    pdf.setTextColor(...TABLE_HEADER_FG);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(7);
    pdf.rect(tableX, y0, metricColW, headerH * 2, 'F');
    pdf.text('Metric', tableX + metricColW / 2, y0 + headerH, { align: 'center' });

    // Segment super-headers
    for (let si = 0; si < nSeg; si++) {
      const splitId = splitIds[si];
      const group = segGroups.get(splitId);
      const segName = group ? group[0].name : splitId;
      const x = tableX + metricColW + si * nAth * colW;
      const w = nAth * colW;
      pdf.setFillColor(60, 60, 60);
      pdf.setTextColor(255, 255, 255);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(6.5);
      pdf.rect(x, y0, w, headerH, 'F');
      pdf.text(segName.slice(0, 20), x + w / 2, y0 + headerH * 0.7, { align: 'center' });
    }

    // Athlete sub-headers
    const y1 = y0 + headerH;
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(6.5);
    for (let si = 0; si < nSeg; si++) {
      for (let ai = 0; ai < nAth; ai++) {
        const aname = athleteNamesOrdered[ai];
        const x = tableX + metricColW + (si * nAth + ai) * colW;
        const [ar, ag, ab] = hexToRgb(athleteColorMap[aname] || '#888888');
        pdf.setFillColor(ar, ag, ab);
        pdf.rect(x, y1, colW, headerH, 'F');
        pdf.setTextColor(255, 255, 255);
        pdf.text(aname.slice(0, 12), x + colW / 2, y1 + headerH * 0.7, { align: 'center' });
      }
    }

    // Data rows
    let yRow = y1 + headerH;
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(6.5);
    for (let mi = 0; mi < metricDefs.length; mi++) {
      const [label, statKey, subKey, decimals, goldKey] = metricDefs[mi];

      if (mi % 2 === 0) {
        pdf.setFillColor(...TABLE_ALT_BG);
        pdf.rect(tableX, yRow, metricColW + nDataCols * colW, rowH, 'F');
      }
      pdf.setFillColor(...TABLE_METRIC_BG);
      pdf.rect(tableX, yRow, metricColW, rowH, 'F');

      pdf.setTextColor(34, 34, 34);
      pdf.text(label, tableX + metricColW / 2, yRow + rowH * 0.7, { align: 'center' });

      for (let si = 0; si < nSeg; si++) {
        const splitId = splitIds[si];
        for (let ai = 0; ai < nAth; ai++) {
          const aname = athleteNamesOrdered[ai];
          const x = tableX + metricColW + (si * nAth + ai) * colW;
          const segData = segLookup[`${splitId}||${aname}`];
          const cellText = formatMetricCell(segData, metricDefs[mi]);

          const dc = darken(hexToRgb(athleteColorMap[aname] || '#888888'));
          pdf.setTextColor(dc[0], dc[1], dc[2]);
          pdf.text(cellText, x + colW / 2, yRow + rowH * 0.7, { align: 'center' });

          if (segData && isGold(golds, goldKey, splitId, aname)) {
            _drawGoldDot(pdf, x + colW - 2.0, yRow + rowH / 2, 0.9);
          }
        }
      }
      yRow += rowH;
    }

    pdf.setY(yRow + (chunkIdx < splitChunks.length - 1 ? 4 : 2));
  }
}

// ── Segment stats table (matches Python _draw_segment_stats_table) ────

function drawSegmentStatsTable(pdf, segGroup, golds, athleteColorMap, margin, metricDefs = METRIC_DEFS) {
  const athNames = [];
  for (const s of segGroup) {
    if (!athNames.includes(s.athlete_name)) athNames.push(s.athlete_name);
  }
  const nAth = athNames.length;
  if (nAth === 0) return;

  const epw = pdf.internal.pageSize.getWidth() - 2 * margin;
  const metricColW = 36;
  const athColW = Math.min((epw - metricColW) / Math.max(nAth, 1), 50);
  const totalW = metricColW + nAth * athColW;
  const tableX = margin + (epw - totalW) / 2;
  const rowH = 6.5 * PDF_FONT_SCALE;
  const splitId = segGroup[0].split_id;

  // Segment data lookup
  const segByAth = {};
  for (const s of segGroup) segByAth[s.athlete_name] = s;

  const allMetrics = [...metricDefs];

  // ── Header row ────────────────────────────────────────────────
  let y0 = pdf.getY();
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7.5);

  // Metric column header
  pdf.setFillColor(...TABLE_METRIC_HEADER_BG);
  pdf.setTextColor(...TABLE_HEADER_FG);
  pdf.rect(tableX, y0, metricColW, rowH, 'F');
  pdf.text('Metric', tableX + metricColW / 2, y0 + rowH * 0.7, { align: 'center' });

  // Athlete column headers
  for (let ai = 0; ai < nAth; ai++) {
    const aname = athNames[ai];
    const x = tableX + metricColW + ai * athColW;
    const [ar, ag, ab] = hexToRgb(athleteColorMap[aname] || '#888888');
    pdf.setFillColor(ar, ag, ab);
    pdf.rect(x, y0, athColW, rowH, 'F');
    pdf.setTextColor(255, 255, 255);
    pdf.text(aname.slice(0, 15), x + athColW / 2, y0 + rowH * 0.7, { align: 'center' });
  }

  // ── Data rows ─────────────────────────────────────────────────
  let yRow = y0 + rowH;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(7.5);

  for (let mi = 0; mi < allMetrics.length; mi++) {
    const [label, statKey, subKey, decimals, goldKey] = allMetrics[mi];

    if (yRow + rowH > pdf.internal.pageSize.getHeight() - 20) {
      pdf.addPage();
      yRow = margin;
    }

    // Alternating background
    if (mi % 2 === 0) {
      pdf.setFillColor(...TABLE_ALT_BG);
      pdf.rect(tableX, yRow, metricColW + nAth * athColW, rowH, 'F');
    }
    pdf.setFillColor(...TABLE_METRIC_BG);
    pdf.rect(tableX, yRow, metricColW, rowH, 'F');

    // Metric label
    pdf.setTextColor(34, 34, 34);
    pdf.text(label, tableX + metricColW / 2, yRow + rowH * 0.7, { align: 'center' });

    // Athlete values
    for (let ai = 0; ai < nAth; ai++) {
      const aname = athNames[ai];
      const x = tableX + metricColW + ai * athColW;
      const segData = segByAth[aname];
      const cellText = formatMetricCell(segData, allMetrics[mi]);

      // Athlete-coloured text (darkened 30%)
      const dc = darken(hexToRgb(athleteColorMap[aname] || '#888888'));
      pdf.setTextColor(dc[0], dc[1], dc[2]);
      pdf.text(cellText, x + athColW / 2, yRow + rowH * 0.7, { align: 'center' });

      // Gold dot
      if (segData && isGold(golds, goldKey, splitId, aname)) {
        _drawGoldDot(pdf, x + athColW - 2.5, yRow + rowH / 2, 1.0);
      }
    }

    yRow += rowH;
  }

  pdf.setY(yRow + 2);
}

function _drawGoldDot(pdf, x, y, r = 1.0) {
  pdf.setFillColor(...GOLD_COLOR);
  pdf.setDrawColor(...GOLD_COLOR);
  // Use ellipse (always available in jsPDF) as fallback for circle
  if (typeof pdf.circle === 'function') {
    pdf.circle(x, y, r, 'FD');
  } else {
    pdf.ellipse(x, y, r, r, 'FD');
  }
}

// ── Histogram building (matches Python _build_segment_histograms) ─────

function buildSegmentHistograms(group, athleteColorMap, opts = {}) {
  const charts = [];
  const includeBoomAngle = opts?.includeBoomAngle !== false;
  const configs = [
    { title: 'Trunk Angle Distribution', xlabel: 'Trunk Angle', unit: 'deg',
      dataFn: s => (s.trunk_angle_timeline || []).map(d => d.v) },
    { title: 'Rudder Angle Distribution', xlabel: 'Rudder Angle', unit: 'deg',
      dataFn: s => (s.rudder_timeline || []).map(d => d.v) },
    includeBoomAngle ? { title: 'Boom Angle Distribution', xlabel: 'Boom Angle', unit: 'deg',
      dataFn: s => (s.boom_timeline || []).map(d => d.v) } : null,
    { title: 'Rolling Moment Distribution', xlabel: 'Rolling Moment', unit: 'Nm',
      dataFn: s => (s.moment_timeline || []).map(d => Math.abs(d.v)) },
    { title: 'Heel Angle Distribution', xlabel: 'Heel Angle', unit: 'deg',
      dataFn: s => (s.heel_timeline || []).map(d => d.v) },
    { title: 'SOG Distribution', xlabel: 'Speed Over Ground', unit: 'kts',
      dataFn: s => (s.sog_timeline || []).map(d => d.v) },
  ].filter(Boolean);

  for (const cfg of configs) {
    const dataByAthlete = group.map(s => ({
      values: cfg.dataFn(s).filter(v => v != null && isFinite(v)),
      label: s.athlete_name,
      color: athleteColorMap[s.athlete_name] || athleteColorMap[s.athlete_id] || PALETTE[0],
    })).filter(d => d.values.length > 0);

    if (dataByAthlete.length === 0) continue;
    charts.push(renderKdeHistogram(dataByAthlete, null, cfg.xlabel, cfg.title));
  }

  return charts;
}

// ── Embed histogram 2×2 grid (matches Python _embed_histogram_grid) ───

function embedHistogramGrid(pdf, charts, margin) {
  if (charts.length === 0) return;

  const epw = pdf.internal.pageSize.getWidth() - 2 * margin;
  const cellW = epw / 2 - 2;
  const cellH = cellW * (360 / 660); // preserve canvas aspect ratio

  let yRowStart = pdf.getY();

  for (let i = 0; i < charts.length; i++) {
    const col = i % 2;
    if (col === 0) {
      if (pdf.getY() + cellH + 5 > pdf.internal.pageSize.getHeight() - 20) {
        pdf.addPage();
      }
      yRowStart = pdf.getY();
    }

    const x = margin + col * (cellW + 4);
    addImageDataUrl(pdf, charts[i], x, yRowStart, cellW, cellH);

    if (col === 1) {
      pdf.setY(yRowStart + cellH + 3);
    }
  }

  // Advance past last row if odd count
  if (charts.length % 2 === 1) {
    pdf.setY(yRowStart + cellH + 3);
  }
}

// ── Tack analysis timelines ────────────────────────────────────────────

function _niceYRange(values, fallback = [0, 1]) {
  const vals = (values || []).filter(v => Number.isFinite(v));
  if (!vals.length) return [fallback[0], fallback[1]];
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (!(hi > lo)) {
    const pad = Math.max(0.5, Math.abs(lo) * 0.1);
    lo -= pad;
    hi += pad;
  } else {
    const pad = (hi - lo) * 0.12;
    lo -= pad;
    hi += pad;
  }
  return [lo, hi];
}

function renderTackTimelineChart(tack, athleteColor = PALETTE[0]) {
  const W = 1180;
  const H = 980;
  const PAD = { top: 36, right: 28, bottom: 48, left: 92 };
  const ROW_GAP = 10;
  const rows = [
    { key: 'com_y', label: 'COM Y (m)', color: '#1f77b4', fallback: [-0.35, 0.35] },
    { key: 'trunk', label: 'Trunk (deg)', color: '#d62728', fallback: [25, 75] },
    { key: 'rudder', label: 'Rudder (deg)', color: '#ff7f0e', fallback: [-25, 25] },
    { key: 'boom', label: 'Boom (deg)', color: '#17becf', fallback: [-180, 180] },
    { key: 'heading', label: 'COG Delta (deg)', color: '#9467bd', fallback: [-35, 35] },
    { key: 'sog', label: 'SOG (kts)', color: '#2ca02c', fallback: [4, 8] },
  ];

  const { canvas, ctx } = createCanvas(W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const ph = H - PAD.top - PAD.bottom;
  const rowH = (ph - ROW_GAP * (rows.length - 1)) / rows.length;
  const pw = W - PAD.left - PAD.right;

  const allSeries = rows.map(r => Array.isArray(tack?.[r.key]) ? tack[r.key] : []);
  const tackStartRel = 0.0;
  const tackEndRel = Math.max(0.0, Number(tack?.duration_s || 0));

  // Relative timeline scale per tack (plus small padding for readability).
  let dataXMin = Infinity;
  let dataXMax = -Infinity;
  for (const series of allSeries) {
    for (const p of series) {
      const t = Number(p?.t);
      if (!Number.isFinite(t)) continue;
      if (t < dataXMin) dataXMin = t;
      if (t > dataXMax) dataXMax = t;
    }
  }
  const baseXMin = Number.isFinite(dataXMin) ? Math.min(-5.0, dataXMin) : -5.0;
  const baseXMax = Number.isFinite(dataXMax) ? Math.max(tackEndRel + 5.0, dataXMax) : (tackEndRel + 5.0);
  let xMin = baseXMin;
  let xMax = baseXMax;
  if (!(xMax > xMin)) {
    xMin = -5.0;
    xMax = Math.max(5.0, tackEndRel + 5.0);
  }
  const xPad = (xMax - xMin) * 0.04;
  xMin -= xPad;
  xMax += xPad;

  const xToPx = (t) => PAD.left + ((t - xMin) / Math.max(1e-6, (xMax - xMin))) * pw;

  for (let ri = 0; ri < rows.length; ri++) {
    const row = rows[ri];
    const top = PAD.top + ri * (rowH + ROW_GAP);
    const bottom = top + rowH;
    const areaY = bottom - top;
    const series = (Array.isArray(tack?.[row.key]) ? tack[row.key] : [])
      .map(p => ({ t: Number(p?.t), v: Number(p?.v) }))
      .filter(p => Number.isFinite(p.t) && Number.isFinite(p.v));

    // Subtle row background
    ctx.fillStyle = ri % 2 ? 'rgba(248, 249, 251, 0.9)' : 'rgba(255, 255, 255, 1.0)';
    ctx.fillRect(PAD.left, top, pw, areaY);

    // Grid
    ctx.strokeStyle = '#e1e5ea';
    ctx.lineWidth = 0.7;
    ctx.globalAlpha = 0.75;
    for (let gy = 0; gy <= 3; gy++) {
      const y = top + (areaY * gy) / 3;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + pw, y);
      ctx.stroke();
    }
    for (let gx = 0; gx <= 6; gx++) {
      const x = PAD.left + (pw * gx) / 6;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;

    // Shade tack phase (during tack)
    if (tackEndRel > tackStartRel) {
      const sx = xToPx(Math.max(xMin, tackStartRel));
      const ex = xToPx(Math.min(xMax, tackEndRel));
      if (ex > sx) {
        ctx.fillStyle = 'rgba(255, 200, 0, 0.14)';
        ctx.fillRect(sx, top, ex - sx, areaY);
      }
    }

    const [yMin, yMax] = _niceYRange(series.map(p => p.v), row.fallback);
    const yToPx = (v) => bottom - ((v - yMin) / Math.max(1e-6, (yMax - yMin))) * areaY;

    // Axes border
    ctx.strokeStyle = '#222222';
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.rect(PAD.left, top, pw, areaY);
    ctx.stroke();

    // Series
    if (series.length >= 1) {
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.12)';
      ctx.shadowBlur = 1.5;
      ctx.strokeStyle = row.color;
      ctx.lineWidth = 2.1;
      ctx.beginPath();
      for (let i = 0; i < series.length; i++) {
        const x = xToPx(series[i].t);
        const y = yToPx(series[i].v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Y labels
    ctx.fillStyle = '#222222';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ySpan = Math.abs(yMax - yMin);
    const yFmt = ySpan < 1.5 ? 2 : (ySpan < 20 ? 1 : 0);
    ctx.fillText(yMax.toFixed(yFmt), PAD.left - 6, top + 8);
    ctx.fillText(yMin.toFixed(yFmt), PAD.left - 6, bottom - 8);

    // Row label
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = row.color;
    ctx.fillText(row.label, PAD.left + 4, top + 3);
  }

  // Vertical markers for start/end of tack
  const markTimes = [
    { t: tackStartRel, color: '#666666', dash: [6, 3] },
    { t: tackEndRel, color: '#aa7d00', dash: [3, 3] },
  ];
  for (const m of markTimes) {
    const x = xToPx(_clamp(m.t, xMin, xMax));
    ctx.save();
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1.3;
    ctx.setLineDash(m.dash);
    ctx.beginPath();
    ctx.moveTo(x, PAD.top);
    ctx.lineTo(x, H - PAD.bottom);
    ctx.stroke();
    ctx.restore();
  }

  // X axis ticks
  ctx.fillStyle = '#222222';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 6; i++) {
    const t = xMin + (xMax - xMin) * i / 6;
    const x = xToPx(t);
    ctx.fillText(t.toFixed(1), x, H - PAD.bottom + 6);
  }
  ctx.font = '12px sans-serif';
  ctx.fillText('Time From Maneuver Start (s)', PAD.left + pw / 2, H - 18);

  // Header strip
  ctx.fillStyle = athleteColor;
  ctx.globalAlpha = 0.12;
  ctx.fillRect(0, 0, W, 24);
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = '#222222';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const moveLabel = String(tack?.type || '') === 'jibe' ? 'Jibe' : 'Tack';
  const sideFrom = String(tack?.side_from || '?');
  const sideTo = String(tack?.side_to || '?');
  const hdgDelta = Number(tack?.heading_delta_deg);
  const hdgTxt = Number.isFinite(hdgDelta) ? `${hdgDelta.toFixed(1)} deg` : 'n/a';
  ctx.fillText(`${moveLabel} ${sideFrom} -> ${sideTo} | COG change ${hdgTxt} | 5s before / 5s after`, 10, 12);

  return encodeCanvasPng(canvas);
}

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function drawSegmentManeuverAnalysisPages(pdf, segGroup, athleteColorMap, margin, segName, duration, localLabel, localRange) {
  const epw = pdf.internal.pageSize.getWidth() - 2 * margin;
  const pageH = pdf.internal.pageSize.getHeight();

  const maneuverRows = [];
  for (const s of (segGroup || [])) {
    const color = athleteColorMap[s.athlete_name] || athleteColorMap[s.athlete_id] || PALETTE[0];
    const maneuvers = Array.isArray(s?.maneuver_analysis?.moves)
      ? s.maneuver_analysis.moves
      : (Array.isArray(s?.tack_analysis?.tacks) ? s.tack_analysis.tacks : []);
    for (const t of maneuvers) {
      maneuverRows.push({
        athlete: s.athlete_name || 'Athlete',
        color,
        ...t,
      });
    }
  }
  maneuverRows.sort((a, b) => Number(a.start_t || 0) - Number(b.start_t || 0));

  const drawHeader = (continued = false) => {
    pdf.addPage();
    let y = margin;
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(14);
    pdf.setTextColor(34, 34, 34);
    pdf.text(
      `Segment: ${segName} - Maneuver Analysis${continued ? ' (cont.)' : ''}`,
      margin + epw / 2,
      y + 6,
      { align: 'center' }
    );
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(9);
    pdf.setTextColor(80, 80, 80);
    pdf.text(`Duration: ${Math.round(duration)}s`, margin + epw / 2, y + 11, { align: 'center' });
    if (localRange) {
      pdf.text(`${localLabel}: ${localRange}`, margin + epw / 2, y + 15, { align: 'center' });
      y += 19;
    } else {
      y += 15;
    }
    return y;
  };

  let y = drawHeader(false);

  if (!maneuverRows.length) {
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(10);
    pdf.setTextColor(110, 110, 110);
    pdf.text('No tacks or jibes detected in this segment for the selected criteria.', margin + epw / 2, y + 18, { align: 'center' });
    return;
  }

  const chartH = 106;
  const rowGap = 7;
  for (let idx = 0; idx < maneuverRows.length; idx++) {
    const tack = maneuverRows[idx];
    const needed = 7 + chartH + rowGap;
    if (y + needed > pageH - margin) {
      y = drawHeader(true);
    }

    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(10);
    const c = hexToRgb(tack.color || PALETTE[0]);
    pdf.setTextColor(c[0], c[1], c[2]);
    const hdgDelta = Number(tack.heading_delta_deg);
    const t0 = Number(tack.start_t);
    const dt = Number(tack.duration_s);
    const moveLabel = String(tack?.type || '') === 'jibe' ? 'Jibe' : 'Tack';
    pdf.text(
      `${idx + 1}. ${tack.athlete}  |  ${moveLabel}  |  start ${Number.isFinite(t0) ? t0.toFixed(1) : '--'}s  |  duration ${Number.isFinite(dt) ? dt.toFixed(2) : '--'}s  |  dCOG ${Number.isFinite(hdgDelta) ? hdgDelta.toFixed(1) : '--'} deg`,
      margin,
      y + 4
    );
    y += 6;

    const chart = renderTackTimelineChart(tack, tack.color);
    addImageDataUrl(pdf, chart, margin, y, epw, chartH);
    y += chartH + rowGap;
  }
}

// ── Heatmap sections (matches Python _draw_segment_heatmaps) ──────────

async function drawSegmentHeatmaps(pdf, segGroup, athleteColorMap, margin, hullDetail) {
  const epw = pdf.internal.pageSize.getWidth() - 2 * margin;
  const pageH = pdf.internal.pageSize.getHeight();

  // Collect raw XY from report data (same as Python)
  const kpByAthlete = [];
  const comByAthlete = [];

  for (const s of segGroup) {
    const name = s.athlete_name;
    const color = athleteColorMap[name] || athleteColorMap[s.athlete_id] || PALETTE[0];

    if (s.kp_xy?.length > 0) {
      kpByAthlete.push({ points: s.kp_xy, color, label: name });
    }
    if (s.com_xy?.length > 0) {
      comByAthlete.push({ points: s.com_xy, color, label: name });
    }
  }

  if (kpByAthlete.length === 0 && comByAthlete.length === 0) return;

  // Section header
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(12);
  pdf.setTextColor(34, 34, 34);
  pdf.text('Position Heatmaps', margin + epw / 2, pdf.getY() + 5, { align: 'center' });
  pdf.setY(pdf.getY() + 8);

  const rows = [];
  if (kpByAthlete.length > 0) {
    rows.push(await renderHeatmapPerAthlete(kpByAthlete, {
      hullDetail,
      title: 'Keypoint Density',
      layoutCols: 3,
      bowUp: true,
      includeOverlayPanel: true,
    }));
    await yieldHeatmapWork();
  }
  if (comByAthlete.length > 0) {
    rows.push(await renderHeatmapPerAthlete(comByAthlete, {
      grid_center_x: -1.0,
      hullDetail,
      showMeanPos: true,
      title: 'COM Density',
      layoutCols: 3,
      bowUp: true,
      includeOverlayPanel: true,
    }));
    await yieldHeatmapWork();
  }

  const validRows = rows.filter(img => img?.dataUrl && img.width && img.height);
  if (validRows.length === 0) return;

  const rowGap = 4;
  const availH = pageH - 20 - pdf.getY();
  const naturalHeights = validRows.map(img => epw / (img.width / img.height));
  const totalNaturalH = naturalHeights.reduce((a, b) => a + b, 0) + rowGap * (validRows.length - 1);
  let scale = 1;
  if (totalNaturalH > availH && totalNaturalH > 0) {
    scale = Math.max(0.1, (availH - rowGap * (validRows.length - 1)) / naturalHeights.reduce((a, b) => a + b, 0));
  }

  for (let i = 0; i < validRows.length; i++) {
    const img = validRows[i];
    const aspect = img.width / img.height;
    const drawH = naturalHeights[i] * scale;
    const drawW = drawH * aspect;
    const drawX = margin + (epw - drawW) / 2;
    addImageDataUrl(pdf, img.dataUrl, drawX, pdf.getY(), drawW, drawH);
    pdf.setY(pdf.getY() + drawH + (i < validRows.length - 1 ? rowGap : 0));
  }
}

// ── Group helper ──────────────────────────────────────────────────────

function sortSegmentsByRecordingTime(segments) {
  return (segments || []).slice().sort((a, b) => {
    const aStart = Number.isFinite(Number(a?.start_s)) ? Number(a.start_s) : Number.POSITIVE_INFINITY;
    const bStart = Number.isFinite(Number(b?.start_s)) ? Number(b.start_s) : Number.POSITIVE_INFINITY;
    if (aStart !== bStart) return aStart < bStart ? -1 : 1;
    const aEnd = Number.isFinite(Number(a?.end_s)) ? Number(a.end_s) : Number.POSITIVE_INFINITY;
    const bEnd = Number.isFinite(Number(b?.end_s)) ? Number(b.end_s) : Number.POSITIVE_INFINITY;
    if (aEnd !== bEnd) return aEnd < bEnd ? -1 : 1;
    const aName = String(a?.name || '');
    const bName = String(b?.name || '');
    const nameCmp = aName.localeCompare(bName);
    if (nameCmp) return nameCmp;
    const aAthlete = String(a?.athlete_name || '');
    const bAthlete = String(b?.athlete_name || '');
    const athleteCmp = aAthlete.localeCompare(bAthlete);
    if (athleteCmp) return athleteCmp;
    return String(a?.split_id || '').localeCompare(String(b?.split_id || ''));
  });
}

function groupSegmentsBySegId(segments) {
  const map = new Map();
  for (const seg of sortSegmentsByRecordingTime(segments)) {
    if (!map.has(seg.split_id)) map.set(seg.split_id, []);
    map.get(seg.split_id).push(seg);
  }
  for (const [splitId, group] of map.entries()) {
    map.set(splitId, group.slice().sort((a, b) => String(a?.athlete_name || '').localeCompare(String(b?.athlete_name || ''))));
  }
  return map;
}

// ── Main PDF generation (matches Python generate_pdf_report) ──────────

/**
 * Generate a PDF report from report data.
 *
 * @param {object} reportData — from buildReportData()
 * @param {object} [opts]
 * @returns {Blob} — PDF blob
 */
export async function generatePdf(reportData, opts = {}) {
  const reportProgress = (msg, pct) => {
    if (typeof opts?.onProgress === 'function') opts.onProgress(msg, pct);
  };
  const yieldPdfWork = () => new Promise(resolve => setTimeout(resolve, 0));

  reportProgress('Loading PDF engine...', 0.02);
  const JsPDF = await loadJsPdf();
  const pdf = new JsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
  const _origSetFontSize = pdf.setFontSize.bind(pdf);
  pdf.setFontSize = function (v) { return _origSetFontSize(v * PDF_FONT_SCALE); };

  // jsPDF has no built-in cursor tracking like fpdf2 — add getY/setY helpers
  pdf._cursorY = 15;
  pdf.getY = function () { return this._cursorY; };
  pdf.setY = function (v) { this._cursorY = v; };
  const _origAddPage = pdf.addPage.bind(pdf);
  pdf.addPage = function (...args) { _origAddPage(...args); this._cursorY = 15; return this; };

  reportProgress('Loading hull model...', 0.06);
  const hullDetail = await loadHullDetailFromStl();
  reportProgress('Preparing report summary...', 0.10);
  await yieldPdfWork();

  const pageW = pdf.internal.pageSize.getWidth();  // 210
  const margin = 15;
  const epw = pageW - 2 * margin;

  const segments = sortSegmentsByRecordingTime(reportData.segments || []);
  const athletes = reportData.athletes || [];
  const golds = reportData.golds || {};
  const includeSummaryStats = opts?.includeSummaryStats !== false;
  const includeHistograms = opts?.includeHistograms !== false;
  const includeHeatmaps = opts?.includeHeatmaps !== false;
  const includeBoomAngle = opts?.includeBoomAngle !== false;
  const metricDefs = reportMetricDefs({ includeBoomAngle });
  const enablePolarPlots = Boolean(opts?.polarPlots);
  const enableManeuverAnalysis = Boolean(opts?.maneuverAnalysis ?? opts?.tackAnalysis);

  // Build athlete colour map keyed by NAME (like Python).
  // We track which color indices have been used.
  const athleteColorMap = {};
  const _usedColorIdx = new Set();
  for (let i = 0; i < athletes.length; i++) {
    const c = validAthleteColor(athletes[i]?.color) ? athletes[i].color.trim() : athleteColor(i);
    if (athletes[i]?.name) athleteColorMap[athletes[i].name] = c;
    if (athletes[i]?.athlete_id) athleteColorMap[athletes[i].athlete_id] = c;
    if (athletes[i]?.id) athleteColorMap[athletes[i].id] = c;
    if (!validAthleteColor(athletes[i]?.color)) _usedColorIdx.add(i);
  }
  // Also map segment athlete_names into the colour map via their athlete_id.
  // Handles mismatch when athletes[].name differs from segments[].athlete_name.
  for (const seg of segments) {
    const segColor = validAthleteColor(seg?.athlete_color)
      ? seg.athlete_color.trim()
      : (validAthleteColor(seg?.color) ? seg.color.trim() : null);
    if (segColor) {
      if (seg.athlete_name) athleteColorMap[seg.athlete_name] = segColor;
      if (seg.athlete_id) athleteColorMap[seg.athlete_id] = segColor;
      continue;
    }
    if (!athleteColorMap[seg.athlete_name] && athleteColorMap[seg.athlete_id]) {
      athleteColorMap[seg.athlete_name] = athleteColorMap[seg.athlete_id];
    }
    if (!athleteColorMap[seg.athlete_name]) {
      // Fallback: assign a fresh color index
      let idx = 0;
      while (_usedColorIdx.has(idx)) idx++;
      _usedColorIdx.add(idx);
      athleteColorMap[seg.athlete_name] = athleteColor(idx);
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // Page 1: Title + Summary Map + Summary Table
  // ══════════════════════════════════════════════════════════════════

  // Title (matches Python exactly)
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(22);
  pdf.setTextColor(34, 34, 34);
  pdf.text('TrollFish Sailing Report', pageW / 2, 30, { align: 'center' });

  const uniqueSegments = [...new Set(segments.map(s => s.split_id))];
  const browserTz = localTimeZone();
  const sessionTzInfo = await resolveGpsTimeZone(segments, browserTz);
  const sessionTz = sessionTzInfo?.timeZone || browserTz;
  const sessionPlace = sessionTzInfo?.locationLabel || null;
  const gpsSessionStart = Number(reportData?.session_utc_start_s);
  const gpsSessionEnd = Number(reportData?.session_utc_end_s);
  const gpsRange = formatSegmentRangeInZone(gpsSessionStart, gpsSessionEnd, sessionTz, sessionPlace);
  const startCandidates = segments.map(s => Number(s.start_s)).filter(isEpochSeconds);
  const endCandidates = segments.map(s => Number(s.end_s)).filter(isEpochSeconds);
  const hasTimeRange = startCandidates.length > 0 && endCandidates.length > 0;
  const segRange = hasTimeRange
    ? formatSegmentRangeInZone(Math.min(...startCandidates), Math.max(...endCandidates), sessionTz, sessionPlace)
    : null;
  const rangeLine = gpsRange
    ? `Session: ${gpsRange}`
    : (segRange ? `Session: ${segRange}` : null);
  const summaryLine = uniqueSegments.length ? `${uniqueSegments.length} segment(s)` : null;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(9);
  pdf.setTextColor(100, 100, 100);
  if (summaryLine) pdf.text(summaryLine, pageW / 2, 37, { align: 'center' });
  if (rangeLine) {
    pdf.text(rangeLine, pageW / 2, summaryLine ? 42 : 37, { align: 'center' });
  }

  let y = rangeLine ? (summaryLine ? 49 : 44) : (summaryLine ? 44 : 37);

  // Summary map (Python: _make_summary_map_image)
  const summaryMapImg = await renderSummaryMap(
    reportData.full_track_points || {},
    segments,
    athleteColorMap,
    1680, 1080
  );
  if (summaryMapImg) {
    const summaryAspect = 1080 / 1680;
    let mapW = epw;
    let mapH = mapW * summaryAspect;
    const maxSummaryH = 110;
    if (mapH > maxSummaryH) {
      mapH = maxSummaryH;
      mapW = mapH / summaryAspect;
    }
    const mapX = margin + (epw - mapW) / 2;
    addImageDataUrl(pdf, summaryMapImg, mapX, y, mapW, mapH);
    y += mapH + 3;
  }

  pdf.setY(y);

  if (includeSummaryStats) {
    // Summary Statistics heading (matches Python)
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(13);
    pdf.setTextColor(34, 34, 34);
    pdf.text('Summary Statistics', margin + epw / 2, pdf.getY() + 6, { align: 'center' });
    pdf.setY(pdf.getY() + 9);

    // Summary table (Python: _draw_summary_table)
    if (segments.length > 0) {
      drawSummaryTable(pdf, segments, athletes, athleteColorMap, golds, margin, metricDefs);
    }
  }
  reportProgress('Summary pages ready', 0.18);
  await yieldPdfWork();

  // ══════════════════════════════════════════════════════════════════
  // Per-segment pages: map/table page, histogram page, heatmap page
  // ══════════════════════════════════════════════════════════════════

  const segGroups = [...groupSegmentsBySegId(segments).entries()];

  for (let groupIdx = 0; groupIdx < segGroups.length; groupIdx++) {
    const [, group] = segGroups[groupIdx];
    const groupBasePct = 0.18 + (groupIdx / Math.max(1, segGroups.length)) * 0.76;
    const groupSpanPct = 0.76 / Math.max(1, segGroups.length);
    const reportGroupProgress = (msg, frac) => {
      reportProgress(msg, groupBasePct + groupSpanPct * Math.max(0, Math.min(1, frac)));
    };
    const segName = group[0].name;
    const duration = group[0].duration_s;
    const segStart = Number(group[0].start_s);
    const segEnd = Number(group[0].end_s);
    const segTzInfo = await resolveGpsTimeZone(group, sessionTz);
    const segPlace = segTzInfo?.locationLabel || sessionPlace || null;
    const localRange = formatSegmentRangeInZone(segStart, segEnd, segTzInfo?.timeZone || sessionTz, segPlace);
    const localLabel = 'Session';
    reportGroupProgress(`Rendering segment ${segName}...`, 0);
    await yieldPdfWork();

    // Page A: segment map + segment table
    pdf.addPage();
    y = margin;

    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(15);
    pdf.setTextColor(34, 34, 34);
    pdf.text(`Segment: ${segName}`, margin + epw / 2, y + 7, { align: 'center' });
    y += 12;

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(9);
    pdf.setTextColor(80, 80, 80);
    pdf.text(`Duration: ${Math.round(duration)}s`, margin + epw / 2, y + 3, { align: 'center' });
    if (localRange) {
      pdf.text(`${localLabel}: ${localRange}`, margin + epw / 2, y + 7, { align: 'center' });
      y += 12;
    } else {
      y += 8;
    }

    const segMapPaths = group.filter(s => s.gps_path?.length).map(s => ({
      points: s.gps_path,
      color: athleteColorMap[s.athlete_name] || athleteColorMap[s.athlete_id] || PALETTE[0],
      label: s.athlete_name,
    }));
    const segMapImg = await renderGpsMap(segMapPaths, 1560, 840);
    if (segMapImg) {
      const segAspect = 840 / 1560;
      let mapW = epw;
      let mapH = mapW * segAspect;
      const maxSegH = 75;
      if (mapH > maxSegH) {
        mapH = maxSegH;
        mapW = mapH / segAspect;
      }
      const mapX = margin + (epw - mapW) / 2;
      addImageDataUrl(pdf, segMapImg, mapX, y, mapW, mapH);
      y += mapH + 3;
    }
    reportGroupProgress(`Rendering segment ${segName} map...`, 0.22);

    pdf.setY(y);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(10);
    pdf.setTextColor(34, 34, 34);
    if (includeSummaryStats) {
      pdf.text('Segment Statistics', margin + epw / 2, pdf.getY() + 5, { align: 'center' });
      pdf.setY(pdf.getY() + 8);
      drawSegmentStatsTable(pdf, group, golds, athleteColorMap, margin, metricDefs);
    }
    reportGroupProgress(`Rendering segment ${segName} stats...`, 0.38);
    await yieldPdfWork();

    // Page B: histograms
    if (includeHistograms) {
      pdf.addPage();
      y = margin;
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(14);
      pdf.setTextColor(34, 34, 34);
      pdf.text(`Segment: ${segName} - Histograms`, margin + epw / 2, y + 6, { align: 'center' });
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(9);
      pdf.setTextColor(80, 80, 80);
      pdf.text(`Duration: ${Math.round(duration)}s`, margin + epw / 2, y + 11, { align: 'center' });
      if (localRange) {
        pdf.text(`${localLabel}: ${localRange}`, margin + epw / 2, y + 15, { align: 'center' });
        pdf.setY(y + 19);
      } else {
        pdf.setY(y + 15);
      }

      const histCharts = buildSegmentHistograms(group, athleteColorMap, { includeBoomAngle });
      if (histCharts.length > 0) {
        embedHistogramGrid(pdf, histCharts, margin);
      } else {
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(10);
        pdf.setTextColor(110, 110, 110);
        pdf.text('No histogram data available for this segment.', margin + epw / 2, y + 28, { align: 'center' });
      }
    }
    reportGroupProgress(`Rendering segment ${segName} histograms...`, enablePolarPlots ? 0.56 : 0.62);
    await yieldPdfWork();

    // Page C: optional polar plots
    if (enablePolarPlots) {
      drawSegmentPolarPages(
        pdf,
        group,
        athleteColorMap,
        margin,
        segName,
        duration,
        localLabel,
        localRange
      );
      reportGroupProgress(`Rendering segment ${segName} polar plots...`, enableManeuverAnalysis ? 0.78 : 0.84);
      await yieldPdfWork();
    }

    // Next page: heatmaps
    if (includeHeatmaps) {
      pdf.addPage();
      y = margin;
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(14);
      pdf.setTextColor(34, 34, 34);
      pdf.text(`Segment: ${segName} - Heatmaps`, margin + epw / 2, y + 6, { align: 'center' });
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(9);
      pdf.setTextColor(80, 80, 80);
      pdf.text(`Duration: ${Math.round(duration)}s`, margin + epw / 2, y + 11, { align: 'center' });
      if (localRange) {
        pdf.text(`${localLabel}: ${localRange}`, margin + epw / 2, y + 15, { align: 'center' });
        pdf.setY(y + 19);
      } else {
        pdf.setY(y + 15);
      }

      await drawSegmentHeatmaps(pdf, group, athleteColorMap, margin, hullDetail);
      await yieldPdfWork();
    }
    reportGroupProgress(`Rendering segment ${segName} heatmaps...`, enableManeuverAnalysis ? (enablePolarPlots ? 0.9 : 0.84) : 1);

    // Final optional pages: maneuver analysis timelines
    if (enableManeuverAnalysis) {
      drawSegmentManeuverAnalysisPages(
        pdf,
        group,
        athleteColorMap,
        margin,
        segName,
        duration,
        localLabel,
        localRange
      );
      reportGroupProgress(`Rendering segment ${segName} maneuver analysis...`, 1);
      await yieldPdfWork();
    }
  }

  // Return as blob
  reportProgress('Finalizing PDF...', 0.98);
  await yieldPdfWork();
  return pdf.output('blob');
}
