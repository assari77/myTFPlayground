# Neural Network Playground

A zero-dependency browser-based recreation of [TensorFlow Playground](https://playground.tensorflow.org/) built with HTML, CSS, and ES6+ JavaScript.

## Features

- **Neural Network Engine (`src/nn.js`)**: Feedforward MLP with backpropagation, mini-batch SGD, `Tanh` / `ReLU` / `Sigmoid` / `Linear` activations, and `L1` / `L2` regularization.
- **Datasets & Features (`src/datasets.js`)**:
  - Classification: Circle, Exclusive OR (XOR), Gaussian, Spiral
  - Regression: Plane, Multi-Gaussian
  - Input features: $X_1$, $X_2$, $X_1^2$, $X_2^2$, $X_1 X_2$, $\sin(X_1)$, $\sin(X_2)$
- **Interactive Visualization (`src/heatmap.js`, `src/lossChart.js`)**:
  - Live per-neuron mini-heatmaps (hover any neuron to preview its output on the main canvas)
  - Click any weight link or bias square to inspect or edit its value
  - Live training/test loss sparkline and decision boundary heatmap

## Running Locally

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Tests

```bash
npm test
```
