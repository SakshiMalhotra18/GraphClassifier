import * as tf from '@tensorflow/tfjs';

export interface DataPoint {
  x: number;
  y: number;
  forecast?: number;
}

export type Classification = 'good' | 'passable' | 'bad' | 'none';

export const CLASSES: Classification[] = ['good', 'passable', 'bad', 'none'];

// ─── Synthetic Data Generation ───────────────────────────────────────

export const generateSyntheticDataPoints = (noiseLevel: number, type: 'classic' | 'excel' | 'none'): DataPoint[] => {
  const length = 12;
  const data: DataPoint[] = [];

  if (type === 'none') {
    for (let i = 0; i < length; i++) {
      data.push({ x: i, y: Math.random() * 100 + 50 });
    }
    return data;
  }

  const baseTrend = [100, 110, 120, 135, 150, 140, 145, 160, 170, 180, 190, 200];

  for (let i = 0; i < length; i++) {
    const actualNoise = (Math.random() - 0.5) * 6;
    const actualValue = baseTrend[i] + actualNoise;

    const u1 = Math.random() || 0.0001;
    const u2 = Math.random() || 0.0001;
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

    const noise = randStdNormal * noiseLevel;
    data.push({
      x: i,
      y: actualValue,
      forecast: actualValue + noise,
    });
  }
  return data;
};

// ─── Feature Extraction ──────────────────────────────────────────────

export const extractFeatures = (data: DataPoint[]): number[] => {
  const n = data.length;
  if (n === 0) return [0, 0, 0, 0, 0, 0];

  const actuals = data.map((d) => d.y);
  const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
  const varActual = actuals.reduce((a, b) => a + Math.pow(b - meanActual, 2), 0) / n;
  const stdActual = Math.sqrt(varActual);

  const hasForecast = data.every((d) => d.forecast !== undefined) ? 1 : 0;

  if (hasForecast === 0) {
    return [0, 2.0, 0.0, 2.0, stdActual / (meanActual || 1), 0.0];
  }

  const forecasts = data.map((d) => d.forecast!);
  const meanForecast = forecasts.reduce((a, b) => a + b, 0) / n;
  const varForecast = forecasts.reduce((a, b) => a + Math.pow(b - meanForecast, 2), 0) / n;
  const stdForecast = Math.sqrt(varForecast);

  const residuals = data.map((d) => d.forecast! - d.y);
  const meanResidual = residuals.reduce((a, b) => a + b, 0) / n;
  const varResidual = residuals.reduce((a, b) => a + Math.pow(b - meanResidual, 2), 0) / n;
  const stdResidual = Math.sqrt(varResidual);

  const mse = residuals.reduce((a, b) => a + b * b, 0) / n;
  const rmse = Math.sqrt(mse);
  const normRMSE = rmse / (meanActual || 1);

  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += (actuals[i] - meanActual) * (forecasts[i] - meanForecast);
  }
  cov = cov / n;
  const correlation = stdActual > 0 && stdForecast > 0 ? cov / (stdActual * stdForecast) : 0;

  return [
    hasForecast,
    Math.min(normRMSE, 5.0),
    Math.max(-1.0, Math.min(1.0, correlation)),
    Math.min(stdResidual / (meanActual || 1), 5.0),
    Math.min(stdActual / (meanActual || 1), 5.0),
    Math.min(stdForecast / (meanForecast || 1), 5.0),
  ];
};

// ─── Training Dataset ────────────────────────────────────────────────

export const generateTrainingSet = (sizePerClass: number = 100): { features: number[][]; labels: number[][] } => {
  const features: number[][] = [];
  const labels: number[][] = [];

  for (let i = 0; i < sizePerClass; i++) {
    const noise = 1 + Math.random() * 7;
    features.push(extractFeatures(generateSyntheticDataPoints(noise, 'classic')));
    labels.push([1, 0, 0, 0]);
  }

  for (let i = 0; i < sizePerClass; i++) {
    const noise = 18 + Math.random() * 17;
    features.push(extractFeatures(generateSyntheticDataPoints(noise, 'classic')));
    labels.push([0, 1, 0, 0]);
  }

  for (let i = 0; i < sizePerClass; i++) {
    const noise = 55 + Math.random() * 45;
    features.push(extractFeatures(generateSyntheticDataPoints(noise, 'classic')));
    labels.push([0, 0, 1, 0]);
  }

  for (let i = 0; i < sizePerClass; i++) {
    features.push(extractFeatures(generateSyntheticDataPoints(0, 'none')));
    labels.push([0, 0, 0, 1]);
  }

  return { features, labels };
};

// ─── Model Creation ──────────────────────────────────────────────────

export const createModel = (learningRate: number = 0.01): tf.Sequential => {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: 'relu', kernelInitializer: 'varianceScaling' }));
  model.add(tf.layers.dense({ units: 8, activation: 'relu', kernelInitializer: 'varianceScaling' }));
  model.add(tf.layers.dense({ units: 4, activation: 'softmax', kernelInitializer: 'varianceScaling' }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// ─── Model Training ──────────────────────────────────────────────────

export const trainModel = async (
  model: tf.Sequential,
  features: number[][],
  labels: number[][],
  epochs: number = 50,
  batchSize: number = 32,
  onEpochEnd: (epoch: number, logs: any) => void
): Promise<tf.History> => {
  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(labels);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (logs) onEpochEnd(epoch + 1, logs);
      },
    },
  });

  xs.dispose();
  ys.dispose();

  return history;
};

// ─── Prediction ──────────────────────────────────────────────────────

export interface PredictionResult {
  label: Classification;
  confidence: number;
  probabilities: { [key in Classification]: number };
  metrics: { mape: number; rmse: number; correlation: number; noiseRatio: number };
  inferenceTimeMs: number;
}

export const predict = (model: tf.Sequential, data: DataPoint[]): PredictionResult => {
  const t0 = performance.now();
  const features = extractFeatures(data);
  const inputTensor = tf.tensor2d([features]);
  const outputTensor = model.predict(inputTensor) as tf.Tensor;
  const probabilitiesArray = outputTensor.dataSync();
  inputTensor.dispose();
  outputTensor.dispose();
  const inferenceTimeMs = performance.now() - t0;

  let maxIdx = 0;
  let maxVal = 0;
  for (let i = 0; i < probabilitiesArray.length; i++) {
    if (probabilitiesArray[i] > maxVal) {
      maxVal = probabilitiesArray[i];
      maxIdx = i;
    }
  }

  const metrics = computeMetrics(data);

  return {
    label: CLASSES[maxIdx],
    confidence: maxVal,
    probabilities: { good: probabilitiesArray[0], passable: probabilitiesArray[1], bad: probabilitiesArray[2], none: probabilitiesArray[3] },
    metrics,
    inferenceTimeMs,
  };
};

export const predictHeuristic = (data: DataPoint[]): PredictionResult => {
  const t0 = performance.now();
  const metrics = computeMetrics(data);
  let label: Classification = 'none';

  if (data.length > 0 && data.every((d) => d.forecast !== undefined)) {
    if (metrics.noiseRatio < 15) label = 'good';
    else if (metrics.noiseRatio < 45) label = 'passable';
    else label = 'bad';
  }

  const probabilities = {
    good: label === 'good' ? 0.95 : label === 'passable' ? 0.04 : 0.01,
    passable: label === 'passable' ? 0.92 : label === 'good' ? 0.05 : 0.03,
    bad: label === 'bad' ? 0.98 : label === 'passable' ? 0.02 : 0.0,
    none: label === 'none' ? 1.0 : 0.0,
  };

  return {
    label,
    confidence: probabilities[label],
    probabilities,
    metrics,
    inferenceTimeMs: performance.now() - t0,
  };
};

// ─── Shared Metric Computation ───────────────────────────────────────

function computeMetrics(data: DataPoint[]) {
  const n = data.length;
  let mape = 0, rmse = 0, correlation = 0, noiseRatio = 0;

  if (n > 0) {
    const actuals = data.map((d) => d.y);
    const meanActual = actuals.reduce((a, b) => a + b, 0) / n;

    if (data.every((d) => d.forecast !== undefined)) {
      const forecasts = data.map((d) => d.forecast!);
      const residuals = data.map((d) => d.forecast! - d.y);
      const mse = residuals.reduce((a, b) => a + b * b, 0) / n;
      rmse = Math.sqrt(mse);

      let absPercentageSum = 0;
      for (let i = 0; i < n; i++) absPercentageSum += Math.abs(residuals[i]) / (actuals[i] || 1);
      mape = (absPercentageSum / n) * 100;

      const varActual = actuals.reduce((a, b) => a + Math.pow(b - meanActual, 2), 0) / n;
      const stdActual = Math.sqrt(varActual);
      const meanForecast = forecasts.reduce((a, b) => a + b, 0) / n;
      const varForecast = forecasts.reduce((a, b) => a + Math.pow(b - meanForecast, 2), 0) / n;
      const stdForecast = Math.sqrt(varForecast);

      let cov = 0;
      for (let i = 0; i < n; i++) cov += (actuals[i] - meanActual) * (forecasts[i] - meanForecast);
      cov /= n;
      correlation = stdActual > 0 && stdForecast > 0 ? cov / (stdActual * stdForecast) : 0;

      const meanResidual = residuals.reduce((a, b) => a + b, 0) / n;
      const varResidual = residuals.reduce((a, b) => a + Math.pow(b - meanResidual, 2), 0) / n;
      noiseRatio = (Math.sqrt(varResidual) / (stdActual || 1)) * 100;
    }
  }

  return { mape, rmse, correlation, noiseRatio };
}

// ─── IndexedDB Model Caching ─────────────────────────────────────────

const MODEL_KEY = 'indexeddb://graph-classifier-v1';

export const saveModelToCache = async (model: tf.Sequential): Promise<void> => {
  try {
    await model.save(MODEL_KEY);
  } catch (e) {
    console.warn('Failed to cache model to IndexedDB:', e);
  }
};

export const loadModelFromCache = async (): Promise<tf.Sequential | null> => {
  try {
    const model = (await tf.loadLayersModel(MODEL_KEY)) as tf.Sequential;
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  } catch {
    return null;
  }
};

export const hasCachedModel = async (): Promise<boolean> => {
  try {
    const models = await tf.io.listModels();
    return MODEL_KEY in models;
  } catch {
    return false;
  }
};

// ─── Classification Explanation Generator ────────────────────────────

export const getClassificationExplanation = (pred: PredictionResult): string => {
  if (pred.label === 'none') {
    return 'No forecast series detected. The input appears to be a standalone visualization (e.g., bar chart, scatter plot, or blank report) without an actual-vs-forecast comparison structure.';
  }

  const parts: string[] = [];
  const { mape, correlation, noiseRatio } = pred.metrics;

  // MAPE insight
  if (mape > 0) {
    if (mape < 5) parts.push(`very low mean absolute percentage error (${mape.toFixed(1)}%), indicating high accuracy`);
    else if (mape < 15) parts.push(`moderate MAPE of ${mape.toFixed(1)}%, acceptable for general use cases`);
    else parts.push(`elevated MAPE of ${mape.toFixed(1)}%, suggesting significant forecast deviation`);
  }

  // Correlation insight
  if (correlation > 0.95) parts.push(`strong positive correlation (r = ${correlation.toFixed(3)}) — forecast closely tracks actuals`);
  else if (correlation > 0.8) parts.push(`moderate correlation (r = ${correlation.toFixed(3)}) — directional agreement present but with drift`);
  else if (correlation > 0.5) parts.push(`weak correlation (r = ${correlation.toFixed(3)}) — limited directional agreement`);
  else parts.push(`very weak or negative correlation (r = ${correlation.toFixed(3)}) — forecasts do not track actuals`);

  // Noise insight
  if (noiseRatio < 15) parts.push(`low residual noise floor (${noiseRatio.toFixed(1)}%)`);
  else if (noiseRatio < 45) parts.push(`moderate noise detected (${noiseRatio.toFixed(1)}%), manual review advisable`);
  else parts.push(`high noise floor (${noiseRatio.toFixed(1)}%), forecast reliability is compromised`);

  const verdict =
    pred.label === 'good'
      ? 'Forecast meets quality thresholds for automated pipeline use.'
      : pred.label === 'passable'
        ? 'Forecast is borderline — consider manual validation before production dispatch.'
        : 'Forecast quality is below acceptable thresholds. Retrain or recalibrate the upstream model.';

  return `Detected ${parts.join('; ')}. ${verdict}`;
};
