export interface NormalizedEcg {
  signals: Record<string, Float32Array>;
  samplingRate: number;
  sourceKeys: Record<string, string>;
}

export interface RenderEcgOptions {
  dpi?: number;
  widthMm?: number;
  heightMm?: number;
  paperSeconds?: number;
  speedMmPerSecond?: number;
  gainMmPerMv?: number;
  inputUnit?: "mV" | "uV" | "microvolt";
  showGrid?: boolean;
  showCalibration?: boolean;
  showFooter?: boolean;
  showHeader?: boolean;
  showLeadNames?: boolean;
  removeSegmentMedian?: boolean;
  samplingRate?: number;
  metadata?: Record<string, string | number>;
}

export function readNpzEcg(
  source: string | Blob | ArrayBuffer | Uint8Array,
  options?: RenderEcgOptions
): Promise<NormalizedEcg>;

export function renderEcgToCanvas(
  ecgInput: NormalizedEcg | Record<string, unknown>,
  canvas: HTMLCanvasElement,
  options?: RenderEcgOptions
): {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  samplingRate: number;
  renderedLeads: string[];
  rhythmLead: string;
};
