export const DEFAULT_LEAD_COLUMN_GROUPS = Object.freeze([
  Object.freeze(["I", "II", "III"]),
  Object.freeze(["aVR", "aVL", "aVF"]),
  Object.freeze(["V1", "V2", "V3"]),
  Object.freeze(["V4", "V5", "V6"]),
]);

export const STANDARD_12_LEADS = Object.freeze([
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
]);

export const ECG_LAYOUTS = Object.freeze({
  repo3x4: Object.freeze([
    Object.freeze(["I", "aVR", "V1", "V4"]),
    Object.freeze(["II", "aVL", "V2", "V5"]),
    Object.freeze(["III", "aVF", "V3", "V6"]),
  ]),
  groupedRows4x3: DEFAULT_LEAD_COLUMN_GROUPS,
});

const DEFAULT_OPTIONS = Object.freeze({
  dpi: 200,
  widthMm: 279.4, // ECG-image-kit default: 11 inches wide.
  heightMm: 215.9, // ECG-image-kit default: 8.5 inches tall.
  paperSeconds: 10,
  speedMmPerSecond: 25,
  gainMmPerMv: 10,
  inputUnit: "mV",
  showGrid: true,
  showCalibration: true,
  showFooter: true,
  showHeader: false,
  showLeadNames: true,
  removeSegmentMedian: true,
  lineColor: "rgba(18, 18, 18, 0.94)",
  majorGridColor: "rgba(255, 0, 0, 0.58)",
  minorGridColor: "rgba(255, 220, 215, 0.9)",
  textColor: "#111111",
  backgroundColor: "#ffffff",
  traceLineWidthMm: 0.16,
  calibrationLineWidthMm: 0.22,
  separatorLineWidthMm: 0.36,
  fontFamily: "Arial, Helvetica, sans-serif",
  leadFontMm: 3.7,
  footerFontMm: 3.7,
  headerFontMm: 3.9,
});

const textDecoder = new TextDecoder("utf-8");

export async function renderEcgFromNpz(source, canvas, options = {}) {
  const ecg = await readNpzEcg(source, options);
  return renderEcgToCanvas(ecg, canvas, options);
}

export function renderEcgToCanvas(ecgInput, canvas, options = {}) {
  if (!canvas || typeof canvas.getContext !== "function") {
    throw new TypeError("renderEcgToCanvas requires a HTMLCanvasElement.");
  }

  const opts = { ...DEFAULT_OPTIONS, ...options };
  const ecg = normalizeEcgSignals(ecgInput);
  const sampleRate = numberOr(options.samplingRate, ecg.samplingRate, 500);
  const pxPerMm = opts.dpi / 25.4;
  const widthPx = Math.round(opts.widthMm * pxPerMm);
  const heightPx = Math.round(opts.heightMm * pxPerMm);

  canvas.width = widthPx;
  canvas.height = heightPx;
  canvas.style.aspectRatio = `${widthPx} / ${heightPx}`;
  canvas.style.width = canvas.style.width || "100%";
  canvas.style.height = canvas.style.height || "auto";

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, widthPx, heightPx);
  ctx.fillStyle = opts.backgroundColor;
  ctx.fillRect(0, 0, widthPx, heightPx);

  if (opts.showGrid) {
    drawEcgGrid(ctx, opts, pxPerMm);
  }

  if (opts.showHeader) {
    drawHeader(ctx, opts, pxPerMm);
  }

  drawLeadLayout(ctx, ecg, sampleRate, opts, pxPerMm);

  if (opts.showFooter) {
    drawFooter(ctx, opts, pxPerMm);
  }

  ctx.restore();
  return {
    canvas,
    width: widthPx,
    height: heightPx,
    samplingRate: sampleRate,
    renderedLeads: getLayoutRows(opts).flat(),
    rhythmLead: chooseRhythmLead(ecg.signals),
  };
}

export function renderEcgToDataUrl(ecgInput, options = {}) {
  const canvas = document.createElement("canvas");
  renderEcgToCanvas(ecgInput, canvas, options);
  return canvas.toDataURL("image/png");
}

export async function readNpzEcg(source, options = {}) {
  const buffer = await sourceToArrayBuffer(source);
  const entries = await unzipNpz(buffer, options);
  const raw = {};

  for (const [name, bytes] of Object.entries(entries)) {
    if (!name.toLowerCase().endsWith(".npy")) continue;
    const leafName = name.split(/[\\/]/).pop() || name;
    const key = leafName.replace(/\.npy$/i, "");
    raw[key] = parseNpy(bytes);
  }

  return normalizeEcgSignals(raw);
}

export function normalizeEcgSignals(input) {
  if (!input || typeof input !== "object") {
    throw new TypeError("ECG input must be a signal object or normalized ECG object.");
  }

  const rawSignals = input.signals && typeof input.signals === "object" ? input.signals : input;
  const samplingRate =
    scalarNumber(input.samplingRate) ??
    scalarNumber(input.sampling_rate) ??
    scalarNumber(rawSignals.samplingRate) ??
    scalarNumber(rawSignals.sampling_rate);

  const signals = {};
  const sourceKeys = {};

  for (const [key, value] of Object.entries(rawSignals)) {
    const canonical = canonicalSignalKey(key);
    if (!canonical) continue;

    if (canonical === "sampling_rate") {
      continue;
    }

    const array = toFloatSignal(value);
    if (!array || array.length === 0) continue;

    signals[canonical] = array;
    sourceKeys[canonical] = key;
  }

  return {
    signals,
    samplingRate: samplingRate ?? 500,
    sourceKeys,
  };
}

function drawEcgGrid(ctx, opts, pxPerMm) {
  const widthPx = Math.round(opts.widthMm * pxPerMm);
  const heightPx = Math.round(opts.heightMm * pxPerMm);

  // ECG-image-kit uses a 5 mm major grid. At 25 mm/s this is 0.2 s;
  // at 10 mm/mV this is 0.5 mV. Minor boxes are 1 mm.
  for (let mm = 0; mm <= opts.widthMm + 0.01; mm += 1) {
    const isMajor = Math.round(mm) % 5 === 0;
    ctx.beginPath();
    ctx.strokeStyle = isMajor ? opts.majorGridColor : opts.minorGridColor;
    ctx.lineWidth = isMajor ? 0.85 : 0.45;
    const x = mm * pxPerMm;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, heightPx);
    ctx.stroke();
  }

  for (let mm = 0; mm <= opts.heightMm + 0.01; mm += 1) {
    const isMajor = Math.round(mm) % 5 === 0;
    ctx.beginPath();
    ctx.strokeStyle = isMajor ? opts.majorGridColor : opts.minorGridColor;
    ctx.lineWidth = isMajor ? 0.85 : 0.45;
    const y = mm * pxPerMm;
    ctx.moveTo(0, y);
    ctx.lineTo(widthPx, y);
    ctx.stroke();
  }
}

function drawLeadLayout(ctx, ecg, sampleRate, opts, pxPerMm) {
  const signals = ecg.signals;
  const layoutRows = getLayoutRows(opts);
  const columnCount = layoutRows[0]?.length ?? 4;
  const columnSeconds = opts.paperSeconds / columnCount;
  const columnWidthMm = columnSeconds * opts.speedMmPerSecond;
  const leftMarginMm = getRepoStyleLeftMarginMm(opts);
  const calibrationOffsetMm = opts.showCalibration ? 0.2 * opts.speedMmPerSecond : 0;
  const rowHeightMv = getRepoStyleRowHeightMv(opts, layoutRows.length + 1);
  const unitScale = opts.inputUnit === "uV" || opts.inputUnit === "microvolt" ? 0.001 : 1;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (let rowIndex = 0; rowIndex < layoutRows.length; rowIndex += 1) {
    const baselineY = yFromMvAboveBottom(
      opts,
      pxPerMm,
      (layoutRows.length - rowIndex + 0.5) * rowHeightMv
    );

    for (let colIndex = 0; colIndex < columnCount; colIndex += 1) {
      const leadName = layoutRows[rowIndex][colIndex];
      const signal = signals[leadName];
      const timeColumn = getLeadTimeColumn(leadName, opts);
      const segment = resolveColumnSegment(signal, sampleRate, columnSeconds, timeColumn);
      const segmentBaseline = opts.removeSegmentMedian ? finiteMedian(segment) : 0;
      const segmentX =
        (leftMarginMm + colIndex * columnWidthMm + calibrationOffsetMm) * pxPerMm;

      if (opts.showCalibration && colIndex === 0) {
        drawCalibrationPulse(ctx, leftMarginMm * pxPerMm, baselineY, opts, pxPerMm);
      }

      if (opts.showLeadNames) {
        drawLeadLabel(
          ctx,
          leadName,
          segmentX,
          baselineY + 7 * pxPerMm,
          opts,
          pxPerMm
        );
      }

      drawSignalTrace(ctx, segment, {
        baselineY,
        segmentBaseline,
        sampleRate,
        xStart: segmentX,
        pxPerMm,
        opts,
        unitScale,
      });

      if (colIndex < columnCount - 1) {
        const tickX =
          (leftMarginMm +
            colIndex * columnWidthMm +
            calibrationOffsetMm +
            columnSeconds * opts.speedMmPerSecond) *
          pxPerMm;
        drawSeparatorTick(ctx, tickX, baselineY, opts, pxPerMm);
      }
    }
  }

  const rhythmName = chooseRhythmLead(signals);
  const rhythmSignal = signals[rhythmName];
  const rhythmBaselineY = yFromMvAboveBottom(opts, pxPerMm, 0.5 * rowHeightMv + 0.3);
  const rhythmSegment = resolveRhythmSegment(rhythmSignal, sampleRate, opts.paperSeconds);
  const rhythmBaseline = opts.removeSegmentMedian ? finiteMedian(rhythmSegment) : 0;
  const rhythmX = (leftMarginMm + calibrationOffsetMm) * pxPerMm;

  if (opts.showCalibration) {
    drawCalibrationPulse(ctx, leftMarginMm * pxPerMm, rhythmBaselineY, opts, pxPerMm);
  }

  if (opts.showLeadNames) {
    drawLeadLabel(ctx, rhythmName === "RHYTHM" ? "II" : rhythmName, rhythmX, rhythmBaselineY + 7 * pxPerMm, opts, pxPerMm);
  }

  drawSignalTrace(ctx, rhythmSegment, {
    baselineY: rhythmBaselineY,
    segmentBaseline: rhythmBaseline,
    sampleRate,
    xStart: rhythmX,
    pxPerMm,
    opts,
    unitScale,
  });
}

function drawSignalTrace(ctx, samples, state) {
  if (!samples || samples.length === 0) return;

  const { baselineY, segmentBaseline, sampleRate, xStart, pxPerMm, opts, unitScale } = state;
  const pxPerSecond = opts.speedMmPerSecond * pxPerMm;
  const pxPerMv = opts.gainMmPerMv * pxPerMm;

  ctx.beginPath();
  ctx.strokeStyle = opts.lineColor;
  ctx.lineWidth = Math.max(1, opts.traceLineWidthMm * pxPerMm);

  let drawing = false;
  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i];
    if (!Number.isFinite(value)) {
      drawing = false;
      continue;
    }

    const x = xStart + (i / sampleRate) * pxPerSecond;
    const y = baselineY - (value * unitScale - segmentBaseline * unitScale) * pxPerMv;
    if (!drawing) {
      ctx.moveTo(x, y);
      drawing = true;
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

function drawCalibrationPulse(ctx, x, baselineY, opts, pxPerMm) {
  const width = 0.2 * opts.speedMmPerSecond * pxPerMm;
  const height = 1 * opts.gainMmPerMv * pxPerMm;

  // ECG-image-kit draws a 0.2 s, 1 mV DC calibration pulse at the start of
  // each short row and the rhythm strip when calibration is enabled.
  ctx.beginPath();
  ctx.strokeStyle = opts.lineColor;
  ctx.lineWidth = Math.max(1, opts.calibrationLineWidthMm * pxPerMm);
  ctx.moveTo(x, baselineY);
  ctx.lineTo(x, baselineY - height);
  ctx.lineTo(x + width, baselineY - height);
  ctx.lineTo(x + width, baselineY);
  ctx.lineTo(x + width + 1.5 * pxPerMm, baselineY);
  ctx.stroke();
}

function drawSeparatorTick(ctx, x, baselineY, opts, pxPerMm) {
  const halfHeight = 4 * pxPerMm;
  ctx.beginPath();
  ctx.strokeStyle = opts.lineColor;
  ctx.lineWidth = Math.max(1, opts.separatorLineWidthMm * pxPerMm);
  ctx.moveTo(x, baselineY - halfHeight);
  ctx.lineTo(x, baselineY + halfHeight);
  ctx.stroke();
}

function drawLeadLabel(ctx, leadName, x, y, opts, pxPerMm) {
  ctx.fillStyle = opts.textColor;
  ctx.font = `${Math.round(opts.leadFontMm * pxPerMm)}px ${opts.fontFamily}`;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(leadName, x, y);
}

function drawFooter(ctx, opts, pxPerMm) {
  ctx.fillStyle = opts.textColor;
  ctx.font = `${Math.round(opts.footerFontMm * pxPerMm)}px ${opts.fontFamily}`;
  ctx.textBaseline = "alphabetic";
  ctx.fillText("25mm/s", 50 * pxPerMm, (opts.heightMm - 5) * pxPerMm);
  ctx.fillText("10mm/mV", 100 * pxPerMm, (opts.heightMm - 5) * pxPerMm);
}

function drawHeader(ctx, opts, pxPerMm) {
  const metadata = opts.metadata || {};
  const headerRows = [
    [
      `ID: ${metadata.id ?? ""}`,
      `Name: ${metadata.name ?? ""}`,
      `Date: ${metadata.date ?? ""}`,
    ],
    [
      `Age: ${metadata.age ?? ""}`,
      `Height: ${metadata.height ?? ""}`,
      `Weight: ${metadata.weight ?? ""}`,
    ],
    [`Sex: ${metadata.sex ?? ""}`, "", ""],
  ];

  ctx.fillStyle = opts.textColor;
  ctx.font = `${Math.round(opts.headerFontMm * pxPerMm)}px ${opts.fontFamily}`;
  ctx.textBaseline = "top";

  const colX = [1.5, 76, 151].map((mm) => mm * pxPerMm);
  const rowY = [2.5, 8.2, 13.9].map((mm) => mm * pxPerMm);

  for (let row = 0; row < headerRows.length; row += 1) {
    for (let col = 0; col < headerRows[row].length; col += 1) {
      if (headerRows[row][col]) {
        ctx.fillText(headerRows[row][col], colX[col], rowY[row]);
      }
    }
  }
}

function getLayoutRows(opts) {
  if (Array.isArray(opts.layoutRows) && opts.layoutRows.length > 0) {
    return opts.layoutRows;
  }

  const groups = opts.leadColumnGroups || DEFAULT_LEAD_COLUMN_GROUPS;

  // The reference repo stores the standard 3x4 layout as lead column groups:
  // [I, II, III], [aVR, aVL, aVF], [V1, V2, V3], [V4, V5, V6].
  // Visually those groups are transposed into rows:
  // I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6.
  return transpose(groups);
}

function transpose(groups) {
  const rowCount = Math.max(...groups.map((group) => group.length));
  const rows = [];
  for (let row = 0; row < rowCount; row += 1) {
    rows.push(groups.map((group) => group[row]).filter(Boolean));
  }
  return rows;
}

function getLeadTimeColumn(leadName, opts) {
  const groups = opts.leadColumnGroups || DEFAULT_LEAD_COLUMN_GROUPS;
  for (let i = 0; i < groups.length; i += 1) {
    if (groups[i].includes(leadName)) return i;
  }
  return 0;
}

function resolveColumnSegment(signal, sampleRate, columnSeconds, timeColumn) {
  if (!signal || signal.length === 0) return new Float32Array();

  const columnSamples = Math.max(1, Math.round(sampleRate * columnSeconds));
  const shiftedStart = timeColumn * columnSamples;
  const start = signal.length >= shiftedStart + columnSamples ? shiftedStart : 0;
  return signal.slice(start, Math.min(signal.length, start + columnSamples));
}

function resolveRhythmSegment(signal, sampleRate, paperSeconds) {
  if (!signal || signal.length === 0) return new Float32Array();

  const maxSamples = Math.max(1, Math.round(sampleRate * paperSeconds));
  return signal.slice(0, Math.min(signal.length, maxSamples));
}

function chooseRhythmLead(signals) {
  if (signals.RHYTHM && signals.RHYTHM.length) return "RHYTHM";
  if (signals.II && signals.II.length) return "II";
  return STANDARD_12_LEADS.find((lead) => signals[lead]?.length) || "II";
}

function getRepoStyleLeftMarginMm(opts) {
  const xMaxSeconds = opts.widthMm / opts.speedMmPerSecond;
  const gapSeconds = Math.floor(((xMaxSeconds - opts.paperSeconds) / 2) / 0.2) * 0.2;
  return Math.max(0, gapSeconds * opts.speedMmPerSecond);
}

function getRepoStyleRowHeightMv(opts, renderedRows) {
  const dataHeightMv = opts.heightMm / opts.gainMmPerMv;
  return dataHeightMv / (renderedRows + 2);
}

function yFromMvAboveBottom(opts, pxPerMm, mvAboveBottom) {
  return (opts.heightMm - mvAboveBottom * opts.gainMmPerMv) * pxPerMm;
}

function finiteMedian(samples) {
  if (!samples || samples.length === 0) return 0;

  const values = [];
  for (let i = 0; i < samples.length; i += 1) {
    if (Number.isFinite(samples[i])) values.push(samples[i]);
  }

  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

function canonicalSignalKey(key) {
  const compact = String(key)
    .replace(/\.npy$/i, "")
    .replace(/[^a-zA-Z0-9]/g, "")
    .toUpperCase();

  if (["SAMPLINGRATE", "SAMPLERATE", "FS", "FREQUENCY"].includes(compact)) {
    return "sampling_rate";
  }

  if (["RHYTHM", "RHYTHMSTRIP", "LONGRHYTHM", "FULLII", "LONGII"].includes(compact)) {
    return "RHYTHM";
  }

  const withoutLeadPrefix = compact.startsWith("LEAD") ? compact.slice(4) : compact;
  const leadMap = {
    I: "I",
    II: "II",
    III: "III",
    AVR: "aVR",
    AVL: "aVL",
    AVF: "aVF",
    V1: "V1",
    V2: "V2",
    V3: "V3",
    V4: "V4",
    V5: "V5",
    V6: "V6",
  };

  return leadMap[withoutLeadPrefix] || null;
}

function toFloatSignal(value) {
  if (value == null) return null;
  if (typeof value === "number") return new Float32Array([value]);
  if (ArrayBuffer.isView(value)) {
    if (value.length === 0) return new Float32Array();
    if (value instanceof Float32Array) return value;
    return Float32Array.from(value);
  }
  if (Array.isArray(value)) {
    return Float32Array.from(value.map((item) => Number(item)));
  }
  return null;
}

function scalarNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (ArrayBuffer.isView(value) && value.length > 0 && Number.isFinite(Number(value[0]))) {
    return Number(value[0]);
  }
  if (Array.isArray(value) && value.length > 0 && Number.isFinite(Number(value[0]))) {
    return Number(value[0]);
  }
  return undefined;
}

function numberOr(...values) {
  for (const value of values) {
    const numeric = scalarNumber(value);
    if (numeric !== undefined) return numeric;
  }
  return undefined;
}

async function sourceToArrayBuffer(source) {
  if (source instanceof ArrayBuffer) return source;
  if (ArrayBuffer.isView(source)) {
    return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
  }
  if (typeof Blob !== "undefined" && source instanceof Blob) {
    return source.arrayBuffer();
  }
  if (typeof source === "string") {
    if (typeof window === "undefined" && !/^(https?:|data:|blob:)/i.test(source)) {
      const fs = await import("node:fs/promises");
      const { fileURLToPath } = await import("node:url");
      const filePath = source.startsWith("file:") ? fileURLToPath(source) : source;
      const fileBytes = await fs.readFile(filePath);
      return fileBytes.buffer.slice(fileBytes.byteOffset, fileBytes.byteOffset + fileBytes.byteLength);
    }

    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`Unable to fetch NPZ file: ${response.status} ${response.statusText}`);
    }
    return response.arrayBuffer();
  }
  throw new TypeError("NPZ source must be a URL, File, Blob, ArrayBuffer, or Uint8Array.");
}

async function unzipNpz(buffer, options = {}) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocdOffset = findEndOfCentralDirectory(view);
  const entryCount = view.getUint16(eocdOffset + 10, true);
  const centralDirectoryOffset = view.getUint32(eocdOffset + 16, true);
  const entries = {};

  let cursor = centralDirectoryOffset;
  for (let i = 0; i < entryCount; i += 1) {
    if (view.getUint32(cursor, true) !== 0x02014b50) {
      throw new Error("Invalid NPZ central directory.");
    }

    const flags = view.getUint16(cursor + 8, true);
    const method = view.getUint16(cursor + 10, true);
    const compressedSize = view.getUint32(cursor + 20, true);
    const uncompressedSize = view.getUint32(cursor + 24, true);
    const nameLength = view.getUint16(cursor + 28, true);
    const extraLength = view.getUint16(cursor + 30, true);
    const commentLength = view.getUint16(cursor + 32, true);
    const localHeaderOffset = view.getUint32(cursor + 42, true);
    const name = textDecoder.decode(bytes.subarray(cursor + 46, cursor + 46 + nameLength));

    if (flags & 1) {
      throw new Error(`Encrypted NPZ entries are not supported: ${name}`);
    }

    if (view.getUint32(localHeaderOffset, true) !== 0x04034b50) {
      throw new Error(`Invalid NPZ local header for ${name}.`);
    }

    const localNameLength = view.getUint16(localHeaderOffset + 26, true);
    const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
    const dataStart = localHeaderOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);

    if (method === 0) {
      entries[name] = compressed.slice();
    } else if (method === 8) {
      entries[name] = await inflateRaw(compressed, uncompressedSize, options);
    } else {
      throw new Error(`Unsupported NPZ compression method ${method} for ${name}.`);
    }

    cursor += 46 + nameLength + extraLength + commentLength;
  }

  return entries;
}

function findEndOfCentralDirectory(view) {
  const minOffset = Math.max(0, view.byteLength - 65557);
  for (let offset = view.byteLength - 22; offset >= minOffset; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) return offset;
  }
  throw new Error("Invalid NPZ file: end of central directory not found.");
}

async function inflateRaw(bytes, uncompressedSize, options) {
  if (typeof options.inflateRaw === "function") {
    return options.inflateRaw(bytes, uncompressedSize);
  }

  const fflate = options.fflate || globalThis.fflate;
  if (fflate?.inflateRawSync) {
    return fflate.inflateRawSync(bytes);
  }

  if (typeof DecompressionStream !== "undefined") {
    for (const format of ["deflate-raw", "deflate"]) {
      try {
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
        return new Uint8Array(await new Response(stream).arrayBuffer());
      } catch {
        // Try the next browser-supported deflate wrapper.
      }
    }
  }

  try {
    const mod = await import("fflate");
    return mod.inflateRawSync(bytes);
  } catch {
    throw new Error(
      "This NPZ is compressed. Install fflate or pass { inflateRaw } to readNpzEcg()."
    );
  }
}

function parseNpy(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (
    u8[0] !== 0x93 ||
    u8[1] !== 0x4e ||
    u8[2] !== 0x55 ||
    u8[3] !== 0x4d ||
    u8[4] !== 0x50 ||
    u8[5] !== 0x59
  ) {
    throw new Error("Invalid NPY payload inside NPZ.");
  }

  const major = u8[6];
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLength = major <= 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major <= 1 ? 10 : 12;
  const header = textDecoder.decode(u8.subarray(headerStart, headerStart + headerLength));
  const meta = parseNpyHeader(header);
  const dataOffset = headerStart + headerLength;
  const count = meta.shape.length === 0 ? 1 : meta.shape.reduce((total, size) => total * size, 1);
  const array = new Float32Array(count);
  const littleEndian = meta.descr[0] !== ">";
  const type = meta.descr.replace(/[<>=|]/g, "");
  const kind = type[0];
  const byteSize = Number(type.slice(1));

  for (let i = 0; i < count; i += 1) {
    const offset = dataOffset + i * byteSize;
    array[i] = readNpyNumber(view, offset, kind, byteSize, littleEndian);
  }

  return count === 1 ? array[0] : array;
}

function parseNpyHeader(header) {
  const descrMatch = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  const fortranMatch = header.match(/['"]fortran_order['"]\s*:\s*(True|False)/);
  const shapeMatch = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);

  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error("Unsupported NPY header.");
  }

  const shape = shapeMatch[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part));

  return {
    descr: descrMatch[1],
    fortranOrder: fortranMatch[1] === "True",
    shape,
  };
}

function readNpyNumber(view, offset, kind, byteSize, littleEndian) {
  if (kind === "f" && byteSize === 4) return view.getFloat32(offset, littleEndian);
  if (kind === "f" && byteSize === 8) return view.getFloat64(offset, littleEndian);
  if (kind === "i" && byteSize === 1) return view.getInt8(offset);
  if (kind === "i" && byteSize === 2) return view.getInt16(offset, littleEndian);
  if (kind === "i" && byteSize === 4) return view.getInt32(offset, littleEndian);
  if (kind === "i" && byteSize === 8) return Number(view.getBigInt64(offset, littleEndian));
  if (kind === "u" && byteSize === 1) return view.getUint8(offset);
  if (kind === "u" && byteSize === 2) return view.getUint16(offset, littleEndian);
  if (kind === "u" && byteSize === 4) return view.getUint32(offset, littleEndian);
  if (kind === "u" && byteSize === 8) return Number(view.getBigUint64(offset, littleEndian));
  if (kind === "b" && byteSize === 1) return view.getUint8(offset) ? 1 : 0;
  throw new Error(`Unsupported NPY dtype ${kind}${byteSize}.`);
}
