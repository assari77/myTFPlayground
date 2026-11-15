/**
 * Heatmap and scatter plot renderer using the classic diverging orange-white-blue colormap.
 */

const DOMAIN = [-6, 6];

/**
 * Maps a value in [-1, 1] to [r, g, b].
 * -1 -> #f59322 (245, 147, 34)
 *  0 -> #ffffff (255, 255, 255)
 * +1 -> #0877bd (8, 119, 189)
 */
export function valueToRgb(value) {
  const v = Math.max(-1, Math.min(1, value));
  if (v < 0) {
    const t = -v; // 0..1
    return [
      Math.round(255 + (245 - 255) * t),
      Math.round(255 + (147 - 255) * t),
      Math.round(255 + (34 - 255) * t),
    ];
  } else {
    const t = v; // 0..1
    return [
      Math.round(255 + (8 - 255) * t),
      Math.round(255 + (119 - 255) * t),
      Math.round(255 + (189 - 255) * t),
    ];
  }
}

export function valueToCssColor(value) {
  const [r, g, b] = valueToRgb(value);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Renders a 2D scalar field onto a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {Float32Array} values - row-major grid of size gridSize * gridSize
 * @param {number} gridSize
 * @param {object} options - { discretize: boolean, trainData: Array, testData: Array, showTest: boolean }
 */
export function renderHeatmap(canvas, values, gridSize, options = {}) {
  const {
    discretize = false,
    trainData = null,
    testData = null,
    showTest = false,
  } = options;

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  // Draw heatmap into an offscreen ImageData of gridSize x gridSize, then scale smoothly
  const offscreen =
    canvas._offscreen || (canvas._offscreen = document.createElement("canvas"));
  if (offscreen.width !== gridSize || offscreen.height !== gridSize) {
    offscreen.width = gridSize;
    offscreen.height = gridSize;
  }
  const offCtx = offscreen.getContext("2d");
  const imgData = offCtx.createImageData(gridSize, gridSize);
  const data = imgData.data;

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      // Note: canvas y=0 is top (+6), y=gridSize-1 is bottom (-6)
      const val = values[row * gridSize + col];
      const mapped = discretize ? (val >= 0 ? 0.85 : -0.85) : val;
      const [r, g, b] = valueToRgb(mapped);
      const idx = (row * gridSize + col) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  offCtx.putImageData(imgData, 0, 0);

  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = !discretize;
  ctx.drawImage(offscreen, 0, 0, width, height);

  if (trainData || (showTest && testData)) {
    drawAxes(ctx, width, height);
  }

  const toCanvasX = (x) => ((x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * width;
  const toCanvasY = (y) => ((DOMAIN[1] - y) / (DOMAIN[1] - DOMAIN[0])) * height;

  if (showTest && testData) {
    for (let i = 0; i < testData.length; i++) {
      const p = testData[i];
      drawPoint(ctx, toCanvasX(p.x), toCanvasY(p.y), p.label, true);
    }
  }

  if (trainData) {
    for (let i = 0; i < trainData.length; i++) {
      const p = trainData[i];
      drawPoint(ctx, toCanvasX(p.x), toCanvasY(p.y), p.label, false);
    }
  }
}

function drawAxes(ctx, width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  // Horizontal axis (y = 0)
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  // Vertical axis (x = 0)
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();
  ctx.restore();
}

function drawPoint(ctx, cx, cy, label, isTest) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, isTest ? 3.5 : 4, 0, Math.PI * 2);
  ctx.fillStyle = valueToCssColor(label);
  ctx.fill();
  ctx.lineWidth = isTest ? 1.5 : 1.25;
  ctx.strokeStyle = isTest ? "#1f2937" : "#ffffff";
  ctx.stroke();
  ctx.restore();
}

/**
 * Evaluates a function fn(x, y) over a gridSize x gridSize domain in [-6, 6].
 * Row 0 corresponds to y = +6, row gridSize-1 corresponds to y = -6.
 */
export function evaluateGrid(gridSize, fn) {
  const values = new Float32Array(gridSize * gridSize);
  const step = (DOMAIN[1] - DOMAIN[0]) / (gridSize - 1);
  for (let row = 0; row < gridSize; row++) {
    const y = DOMAIN[1] - row * step;
    for (let col = 0; col < gridSize; col++) {
      const x = DOMAIN[0] + col * step;
      values[row * gridSize + col] = fn(x, y);
    }
  }
  return values;
}
