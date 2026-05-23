import * as tf from '@tensorflow/tfjs';

export interface DataPoint {
  x: number;
  y: number;
  forecast?: number;
}

export type Classification = 'good' | 'passable' | 'bad' | 'none';

// Class labels order: 0: good, 1: passable, 2: bad, 3: none
export const CLASSES: Classification[] = ['good', 'passable', 'bad', 'none'];

/**
 * Generates single synthetic time-series dataset.
 */
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
    // Add minor variation to the actual trend to make training diverse
    const actualNoise = (Math.random() - 0.5) * 6;
    const actualValue = baseTrend[i] + actualNoise;
    
    // Box-Muller transform for normal distribution
    const u1 = Math.random() || 0.0001;
    const u2 = Math.random() || 0.0001;
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    
    const noise = randStdNormal * noiseLevel;
    data.push({
      x: i,
      y: actualValue,
      forecast: actualValue + noise
    });
  }
  return data;
};

/**
 * Extracts statistical features from a time series data array.
 * Returns a 6-element feature vector normalized for training.
 */
export const extractFeatures = (data: DataPoint[]): number[] => {
  const n = data.length;
  if (n === 0) return [0, 0, 0, 0, 0, 0];

  const actuals = data.map(d => d.y);
  const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
  
  // Variance & Std of actuals
  const varActual = actuals.reduce((a, b) => a + Math.pow(b - meanActual, 2), 0) / n;
  const stdActual = Math.sqrt(varActual);

  const hasForecast = data.every(d => d.forecast !== undefined) ? 1 : 0;

  if (hasForecast === 0) {
    // If no forecast, return feature vector indicating no forecast
    return [0, 2.0, 0.0, 2.0, stdActual / (meanActual || 1), 0.0];
  }

  const forecasts = data.map(d => d.forecast!);
  const meanForecast = forecasts.reduce((a, b) => a + b, 0) / n;
  const varForecast = forecasts.reduce((a, b) => a + Math.pow(b - meanForecast, 2), 0) / n;
  const stdForecast = Math.sqrt(varForecast);

  // Residuals (noise)
  const residuals = data.map(d => d.forecast! - d.y);
  const meanResidual = residuals.reduce((a, b) => a + b, 0) / n;
  
  // Standard deviation of residuals
  const residualsDiffMean = residuals.map(r => r - meanResidual);
  const varResidual = residualsDiffMean.reduce((a, b) => a + Math.pow(b, 2), 0) / n;
  const stdResidual = Math.sqrt(varResidual);

  // RMSE
  const mse = residuals.reduce((a, b) => a + b * b, 0) / n;
  const rmse = Math.sqrt(mse);
  const normRMSE = rmse / (meanActual || 1);

  // Pearson Correlation Coefficient
  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += (actuals[i] - meanActual) * (forecasts[i] - meanForecast);
  }
  cov = cov / n;
  const correlation = (stdActual > 0 && stdForecast > 0) ? cov / (stdActual * stdForecast) : 0;

  // Normalized features:
  // [hasForecast, normRMSE, correlation, normResidualStd, normActualStd, normForecastStd]
  return [
    hasForecast,
    Math.min(normRMSE, 5.0), // Cap normalized RMSE to prevent spikes
    Math.max(-1.0, Math.min(1.0, correlation)),
    Math.min(stdResidual / (meanActual || 1), 5.0),
    Math.min(stdActual / (meanActual || 1), 5.0),
    Math.min(stdForecast / (meanForecast || 1), 5.0)
  ];
};

/**
 * Generates complete training dataset for the browser model.
 */
export const generateTrainingSet = (sizePerClass: number = 100): { features: number[][], labels: number[][] } => {
  const features: number[][] = [];
  const labels: number[][] = [];

  // 1. Good: noiseLevel 1 to 8 (Standard error relative to actual is very small)
  for (let i = 0; i < sizePerClass; i++) {
    const noise = 1 + Math.random() * 7;
    const pts = generateSyntheticDataPoints(noise, 'classic');
    features.push(extractFeatures(pts));
    labels.push([1, 0, 0, 0]);
  }

  // 2. Passable: noiseLevel 18 to 35
  for (let i = 0; i < sizePerClass; i++) {
    const noise = 18 + Math.random() * 17;
    const pts = generateSyntheticDataPoints(noise, 'classic');
    features.push(extractFeatures(pts));
    labels.push([0, 1, 0, 0]);
  }

  // 3. Bad: noiseLevel 55 to 100
  for (let i = 0; i < sizePerClass; i++) {
    const noise = 55 + Math.random() * 45;
    const pts = generateSyntheticDataPoints(noise, 'classic');
    features.push(extractFeatures(pts));
    labels.push([0, 0, 1, 0]);
  }

  // 4. None: type 'none' (distractor graphs without forecast)
  for (let i = 0; i < sizePerClass; i++) {
    const pts = generateSyntheticDataPoints(0, 'none');
    features.push(extractFeatures(pts));
    labels.push([0, 0, 0, 1]);
  }

  return { features, labels };
};

/**
 * Creates and compiles a sequential neural network model in TF.js.
 */
export const createModel = (learningRate: number = 0.01): tf.Sequential => {
  const model = tf.sequential();
  
  // Input: 6 features
  model.add(tf.layers.dense({
    inputShape: [6],
    units: 16,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  
  // Hidden Layer
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  
  // Output Layer: 4 classes (good, passable, bad, none)
  model.add(tf.layers.dense({
    units: 4,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling'
  }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
};

/**
 * Trains the model with custom progress callback.
 */
export const trainModel = async (
  model: tf.Sequential,
  features: number[][],
  labels: number[][],
  epochs: number = 50,
  batchSize: number = 32,
  onEpochEnd: (epoch: number, logs: any) => void
): Promise<tf.History> => {
  // Convert arrays to tensors
  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(labels);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (logs) {
          onEpochEnd(epoch + 1, logs);
        }
      }
    }
  });

  // Clean up tensors from memory
  xs.dispose();
  ys.dispose();

  return history;
};

/**
 * Runs predictions on given data series.
 */
export const predict = (model: tf.Sequential, data: DataPoint[]): {
  label: Classification;
  confidence: number;
  probabilities: { [key in Classification]: number };
  metrics: {
    mape: number;
    rmse: number;
    correlation: number;
    noiseRatio: number;
  };
} => {
  const features = extractFeatures(data);
  const inputTensor = tf.tensor2d([features]);
  
  const outputTensor = model.predict(inputTensor) as tf.Tensor;
  const probabilitiesArray = outputTensor.dataSync();
  
  inputTensor.dispose();
  outputTensor.dispose();

  // Find class with highest probability
  let maxIdx = 0;
  let maxVal = 0;
  for (let i = 0; i < probabilitiesArray.length; i++) {
    if (probabilitiesArray[i] > maxVal) {
      maxVal = probabilitiesArray[i];
      maxIdx = i;
    }
  }

  const label = CLASSES[maxIdx];
  const confidence = maxVal;

  const probabilities = {
    good: probabilitiesArray[0],
    passable: probabilitiesArray[1],
    bad: probabilitiesArray[2],
    none: probabilitiesArray[3]
  };

  // Compute stats for audit metrics
  const n = data.length;
  let mape = 0;
  let rmse = 0;
  let correlation = 0;
  let noiseRatio = 0;

  if (n > 0) {
    const actuals = data.map(d => d.y);
    const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
    
    if (data.every(d => d.forecast !== undefined)) {
      const forecasts = data.map(d => d.forecast!);
      const residuals = data.map(d => d.forecast! - d.y);
      
      const mse = residuals.reduce((a, b) => a + b * b, 0) / n;
      rmse = Math.sqrt(mse);
      
      let absPercentageSum = 0;
      for (let i = 0; i < n; i++) {
        absPercentageSum += Math.abs(residuals[i]) / (actuals[i] || 1);
      }
      mape = (absPercentageSum / n) * 100;
      
      // Std devs
      const varActual = actuals.reduce((a, b) => a + Math.pow(b - meanActual, 2), 0) / n;
      const stdActual = Math.sqrt(varActual);
      
      const meanForecast = forecasts.reduce((a, b) => a + b, 0) / n;
      const varForecast = forecasts.reduce((a, b) => a + Math.pow(b - meanForecast, 2), 0) / n;
      const stdForecast = Math.sqrt(varForecast);

      let cov = 0;
      for (let i = 0; i < n; i++) {
        cov += (actuals[i] - meanActual) * (forecasts[i] - meanForecast);
      }
      cov = cov / n;
      correlation = (stdActual > 0 && stdForecast > 0) ? cov / (stdActual * stdForecast) : 0;
      
      const meanResidual = residuals.reduce((a, b) => a + b, 0) / n;
      const varResidual = residuals.reduce((a, b) => a + Math.pow(b - meanResidual, 2), 0) / n;
      const stdResidual = Math.sqrt(varResidual);
      
      noiseRatio = (stdResidual / (stdActual || 1)) * 100;
    }
  }

  return {
    label,
    confidence,
    probabilities,
    metrics: {
      mape,
      rmse,
      correlation,
      noiseRatio
    }
  };
};

/**
 * Falling back to a simple rule-based model if TFJS is not loaded or training is skipped.
 */
export const predictHeuristic = (data: DataPoint[]): ReturnType<typeof predict> => {
  const n = data.length;
  let mape = 0;
  let rmse = 0;
  let correlation = 0;
  let noiseRatio = 0;
  let label: Classification = 'none';

  if (n > 0) {
    const actuals = data.map(d => d.y);
    const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
    const hasForecast = data.every(d => d.forecast !== undefined);

    if (hasForecast) {
      const forecasts = data.map(d => d.forecast!);
      const residuals = data.map(d => d.forecast! - d.y);
      const mse = residuals.reduce((a, b) => a + b * b, 0) / n;
      rmse = Math.sqrt(mse);
      
      let absPercentageSum = 0;
      for (let i = 0; i < n; i++) {
        absPercentageSum += Math.abs(residuals[i]) / (actuals[i] || 1);
      }
      mape = (absPercentageSum / n) * 100;
      
      const varActual = actuals.reduce((a, b) => a + Math.pow(b - meanActual, 2), 0) / n;
      const stdActual = Math.sqrt(varActual);
      
      const meanForecast = forecasts.reduce((a, b) => a + b, 0) / n;
      const varForecast = forecasts.reduce((a, b) => a + Math.pow(b - meanForecast, 2), 0) / n;
      const stdForecast = Math.sqrt(varForecast);

      let cov = 0;
      for (let i = 0; i < n; i++) {
        cov += (actuals[i] - meanActual) * (forecasts[i] - meanForecast);
      }
      cov = cov / n;
      correlation = (stdActual > 0 && stdForecast > 0) ? cov / (stdActual * stdForecast) : 0;
      
      const meanResidual = residuals.reduce((a, b) => a + b, 0) / n;
      const varResidual = residuals.reduce((a, b) => a + Math.pow(b - meanResidual, 2), 0) / n;
      const stdResidual = Math.sqrt(varResidual);
      
      noiseRatio = (stdResidual / (stdActual || 1)) * 100;

      // Classify based on noiseRatio
      if (noiseRatio < 15) {
        label = 'good';
      } else if (noiseRatio < 45) {
        label = 'passable';
      } else {
        label = 'bad';
      }
    } else {
      label = 'none';
    }
  }

  // Generate simulated probabilities based on the output
  const probabilities = {
    good: label === 'good' ? 0.95 : label === 'passable' ? 0.04 : 0.01,
    passable: label === 'passable' ? 0.92 : label === 'good' ? 0.05 : 0.03,
    bad: label === 'bad' ? 0.98 : label === 'passable' ? 0.02 : 0.00,
    none: label === 'none' ? 1.00 : 0.00
  };

  return {
    label,
    confidence: probabilities[label],
    probabilities,
    metrics: {
      mape,
      rmse,
      correlation,
      noiseRatio
    }
  };
};
