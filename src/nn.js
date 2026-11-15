/**
 * Pure JavaScript Neural Network engine modeled after TensorFlow Playground.
 * Supports configurable hidden layers, activations (tanh, relu, sigmoid, linear),
 * L1/L2 regularization, mini-batch SGD, and per-node activation inspection.
 */

export const Activations = {
  tanh: {
    output: (x) => Math.tanh(x),
    der: (x) => {
      const t = Math.tanh(x);
      return 1 - t * t;
    },
  },
  relu: {
    output: (x) => (x > 0 ? x : 0),
    der: (x) => (x > 0 ? 1 : 0),
  },
  sigmoid: {
    output: (x) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x)))),
    der: (x) => {
      const s = 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));
      return s * (1 - s);
    },
  },
  linear: {
    output: (x) => x,
    der: () => 1,
  },
};

export const Regularization = {
  none: {
    output: () => 0,
    der: () => 0,
  },
  l1: {
    output: (w) => Math.abs(w),
    der: (w) => (w < 0 ? -1 : w > 0 ? 1 : 0),
  },
  l2: {
    output: (w) => 0.5 * w * w,
    der: (w) => w,
  },
};

/**
 * Deterministic-friendly PRNG (Mulberry32) so datasets/weights can be reproducible if needed.
 */
export function createRng(seed = Date.now()) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Link {
  constructor(source, dest, regularization, initZero = false, rng = Math.random) {
    this.id = `${source.id}-${dest.id}`;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    this.isDead = false;
    this.weight = initZero ? 0 : (rng() * 2 - 1) * 0.5;
    this.errorDer = 0;
    this.accErrorDer = 0;
    this.numAccumulatedDers = 0;
  }
}

export class Node {
  constructor(id, activation, initZero = false, rng = Math.random) {
    this.id = id;
    this.activation = activation;
    this.bias = initZero ? 0 : (rng() * 2 - 1) * 0.1;
    this.inputLinks = [];
    this.outputs = [];
    this.totalInput = 0;
    this.output = 0;
    this.outputDer = 0;
    this.inputDer = 0;
    this.accInputDer = 0;
    this.numAccumulatedDers = 0;
  }

  updateOutput() {
    this.totalInput = this.bias;
    for (let i = 0; i < this.inputLinks.length; i++) {
      const link = this.inputLinks[i];
      if (!link.isDead) {
        this.totalInput += link.weight * link.source.output;
      }
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
 * Builds a fully-connected feedforward network.
 * @param {number[]} networkShape - e.g. [numInputs, 4, 2, 1]
 * @param {object} activation - Activations.tanh | relu | sigmoid | linear
 * @param {object} outputActivation - Activations.tanh (classification) | linear (regression)
 * @param {object} regularization - Regularization.none | l1 | l2
 * @param {string[]} inputIds - IDs of active input features
 * @param {boolean} initZero
 * @param {function} rng
 */
export function buildNetwork(
  networkShape,
  activation,
  outputActivation,
  regularization,
  inputIds,
  initZero = false,
  rng = Math.random
) {
  const numLayers = networkShape.length;
  const network = [];

  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const isOutputLayer = layerIdx === numLayers - 1;
    const isInputLayer = layerIdx === 0;
    const currentLayer = [];
    network.push(currentLayer);

    const numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = `${layerIdx}-${i}`;
      if (isInputLayer && inputIds && inputIds[i]) {
        nodeId = inputIds[i];
      }
      const nodeAct = isOutputLayer
        ? outputActivation
        : isInputLayer
        ? Activations.linear
        : activation;

      const node = new Node(nodeId, nodeAct, initZero, rng);
      currentLayer.push(node);

      if (layerIdx >= 1) {
        const prevLayer = network[layerIdx - 1];
        const fanIn = prevLayer.length;
        // Xavier/Glorot or He scale for better convergence
        const scale =
          nodeAct === Activations.relu
            ? Math.sqrt(2 / Math.max(1, fanIn))
            : Math.sqrt(1 / Math.max(1, fanIn));

        for (let j = 0; j < prevLayer.length; j++) {
          const prevNode = prevLayer[j];
          const link = new Link(prevNode, node, regularization, initZero, rng);
          if (!initZero) {
            link.weight = (rng() * 2 - 1) * scale * 1.4;
          }
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }

  return network;
}

export function forwardProp(network, inputs) {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error(
      `Input length (${inputs.length}) does not match input layer size (${inputLayer.length})`
    );
  }
  for (let i = 0; i < inputLayer.length; i++) {
    inputLayer[i].output = inputs[i];
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      layer[i].updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

export function backProp(network, target) {
  const outputNode = network[network.length - 1][0];
  // MSE loss: 0.5 * (output - target)^2 => dE/dOutput = output - target
  outputNode.outputDer = outputNode.output - target;

  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      const node = layer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    for (let i = 0; i < layer.length; i++) {
      const node = layer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) continue;
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }

    if (layerIdx === 1) continue;

    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        const link = node.outputs[j];
        if (!link.isDead) {
          node.outputDer += link.weight * link.dest.inputDer;
        }
      }
    }
  }
}

export function updateWeights(network, learningRate, regularizationRate) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      const node = layer[i];
      if (node.numAccumulatedDers > 0) {
        node.bias -= (learningRate * node.accInputDer) / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }

      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) continue;

        const regDer = link.regularization
          ? link.regularization.der(link.weight)
          : 0;

        if (link.numAccumulatedDers > 0) {
          link.weight -=
            (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          const newWeight =
            link.weight - learningRate * regularizationRate * regDer;

          // L1 weight zeroing
          if (
            link.regularization === Regularization.l1 &&
            link.weight * newWeight < 0
          ) {
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

export function computeLoss(network, data, constructInput) {
  if (!data || data.length === 0) return 0;
  let loss = 0;
  for (let i = 0; i < data.length; i++) {
    const p = data[i];
    const output = forwardProp(network, constructInput(p.x, p.y));
    const diff = output - p.label;
    loss += 0.5 * diff * diff;
  }
  return loss / data.length;
}

export function forEachNode(network, ignoreInputs, accessor) {
  if (!network) return;
  const startLayer = ignoreInputs ? 1 : 0;
  for (let layerIdx = startLayer; layerIdx < network.length; layerIdx++) {
    const layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      accessor(layer[i], layerIdx, i);
    }
  }
}
