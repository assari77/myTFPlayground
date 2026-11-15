/**
 * Dataset generators and feature extractors modeled after TensorFlow Playground.
 * Coordinates span [-6, 6] x [-6, 6]. Labels are -1 or +1 for classification,
 * and continuous in [-1, 1] for regression.
 */

export const FEATURES = {
  x: { label: "X₁", f: (x, y) => x },
  y: { label: "X₂", f: (x, y) => y },
  xSquared: { label: "X₁²", f: (x, y) => x * x },
  ySquared: { label: "X₂²", f: (x, y) => y * y },
  xTimesY: { label: "X₁X₂", f: (x, y) => x * y },
  sinX: { label: "sin(X₁)", f: (x, y) => Math.sin(x) },
  sinY: { label: "sin(X₂)", f: (x, y) => Math.sin(y) },
};

function normalRandom(mean = 0, variance = 1, rng = Math.random) {
  let v1, v2, s;
  do {
    v1 = 2 * rng() - 1;
    v2 = 2 * rng() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1 || s === 0);
  const result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

function randUniform(a, b, rng = Math.random) {
  return rng() * (b - a) + a;
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function shuffle(array, rng = Math.random) {
  let counter = array.length;
  while (counter > 0) {
    const index = Math.floor(rng() * counter);
    counter--;
    const temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
  return array;
}

export function classifyCircleData(numSamples, noise, rng = Math.random) {
  const points = [];
  const radius = 5;
  const getCircleLabel = (p, center) =>
    dist(p, center) < radius * 0.5 ? 1 : -1;

  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(0, radius * 0.5, rng);
    const angle = randUniform(0, 2 * Math.PI, rng);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius, rng) * noise;
    const noiseY = randUniform(-radius, radius, rng) * noise;
    const label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }

  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(radius * 0.7, radius, rng);
    const angle = randUniform(0, 2 * Math.PI, rng);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius, rng) * noise;
    const noiseY = randUniform(-radius, radius, rng) * noise;
    const label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }

  return points;
}

export function classifyXORData(numSamples, noise, rng = Math.random) {
  const points = [];
  const getXORLabel = (p) => (p.x * p.y >= 0 ? 1 : -1);

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5, rng);
    const padding = 0.3;
    x += x > 0 ? padding : -padding;
    let y = randUniform(-5, 5, rng);
    y += y > 0 ? padding : -padding;
    const noiseX = randUniform(-5, 5, rng) * noise;
    const noiseY = randUniform(-5, 5, rng) * noise;
    const label = getXORLabel({ x: x + noiseX, y: y + noiseY });
    points.push({ x, y, label });
  }
  return points;
}

export function classifyTwoGaussData(numSamples, noise, rng = Math.random) {
  const points = [];
  const varianceScale = 0.5 + noise * 3.5; // scales smoothly with noise [0..0.5]
  const variance = 0.5 * varianceScale;

  function genGauss(cx, cy, label) {
    for (let i = 0; i < numSamples / 2; i++) {
      const x = normalRandom(cx, variance, rng);
      const y = normalRandom(cy, variance, rng);
      points.push({ x, y, label });
    }
  }

  genGauss(2, 2, 1);
  genGauss(-2, -2, -1);
  return points;
}

export function classifySpiralData(numSamples, noise, rng = Math.random) {
  const points = [];
  const n = numSamples / 2;

  function genSpiral(deltaT, label) {
    for (let i = 0; i < n; i++) {
      const r = (i / n) * 5;
      const t = (1.75 * i) / n * 2 * Math.PI + deltaT;
      const x = r * Math.sin(t) + randUniform(-1, 1, rng) * noise;
      const y = r * Math.cos(t) + randUniform(-1, 1, rng) * noise;
      points.push({ x, y, label });
    }
  }

  genSpiral(0, 1);
  genSpiral(Math.PI, -1);
  return points;
}

export function regressPlane(numSamples, noise, rng = Math.random) {
  const radius = 6;
  const points = [];
  const getLabel = (x, y) => Math.max(-1, Math.min(1, (x + y) / 10));

  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-radius, radius, rng);
    const y = randUniform(-radius, radius, rng);
    const noiseX = randUniform(-radius, radius, rng) * noise;
    const noiseY = randUniform(-radius, radius, rng) * noise;
    const label = getLabel(x + noiseX, y + noiseY);
    points.push({ x, y, label });
  }
  return points;
}

export function regressGaussian(numSamples, noise, rng = Math.random) {
  const points = [];
  const labelScale = (x) => Math.max(-1, Math.min(1, x));
  const gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1],
  ];

  const getLabel = (x, y) => {
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      const d = dist({ x, y }, { x: cx, y: cy });
      const newLabel = sign * Math.exp(-(d * d) / 4);
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return labelScale(label);
  };

  const radius = 6;
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-radius, radius, rng);
    const y = randUniform(-radius, radius, rng);
    const noiseX = randUniform(-radius, radius, rng) * noise;
    const noiseY = randUniform(-radius, radius, rng) * noise;
    const label = getLabel(x + noiseX, y + noiseY);
    points.push({ x, y, label });
  }
  return points;
}

export const CLASSIFICATION_DATASETS = {
  circle: { name: "Circle", generator: classifyCircleData },
  xor: { name: "Exclusive OR", generator: classifyXORData },
  gauss: { name: "Gaussian", generator: classifyTwoGaussData },
  spiral: { name: "Spiral", generator: classifySpiralData },
};

export const REGRESSION_DATASETS = {
  plane: { name: "Plane", generator: regressPlane },
  regGauss: { name: "Multi Gaussian", generator: regressGaussian },
};

export function generateData(
  datasetKey,
  problemType,
  noisePercent,
  trainRatioPercent,
  numSamples = 500,
  rng = Math.random
) {
  const noise = noisePercent / 100;
  const registry =
    problemType === "regression" ? REGRESSION_DATASETS : CLASSIFICATION_DATASETS;
  const entry = registry[datasetKey] || Object.values(registry)[0];
  const points = entry.generator(numSamples, noise, rng);
  shuffle(points, rng);
  const splitIndex = Math.floor((points.length * trainRatioPercent) / 100);
  return {
    trainData: points.slice(0, splitIndex),
    testData: points.slice(splitIndex),
  };
}
