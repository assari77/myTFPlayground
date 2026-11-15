import {
  Activations,
  Regularization,
  buildNetwork,
  forwardProp,
  backProp,
  updateWeights,
  computeLoss,
  forEachNode,
} from "./nn.js";
import {
  FEATURES,
  CLASSIFICATION_DATASETS,
  REGRESSION_DATASETS,
  generateData,
  shuffle,
} from "./datasets.js";
import { renderHeatmap, evaluateGrid, valueToCssColor } from "./heatmap.js";
import { LossChart } from "./lossChart.js";

const GRID_MAIN = 48;
const GRID_MINI = 24;

const state = {
  isPlaying: false,
  epoch: 0,
  learningRate: 0.03,
  activationKey: "tanh",
  regKey: "none",
  regRate: 0,
  problemType: "classification",
  datasetKey: "gauss",
  lowLossStreak: 0,
  stabilizedEpoch: null,
  trainRatio: 50,
  noise: 0,
  batchSize: 10,
  hiddenLayers: [4, 2],
  activeFeatures: new Set(["x", "y"]),
  showTest: false,
  discretize: false,
  trainData: [],
  testData: [],
  network: null,
  hoveredNodeId: null,
  nodeCanvases: new Map(), // nodeId -> HTMLCanvasElement
  linkPaths: new Map(), // linkId -> SVGPathElement
  biasTags: new Map(), // nodeId -> HTMLElement
  inspectorTarget: null, // { type: 'link'|'bias', ref }
};

// DOM references
const el = {
  btnStart: document.getElementById("btn-start"),
  btnStop: document.getElementById("btn-stop"),
  epochCounter: document.getElementById("epoch-counter"),
  stabilizedBadge: document.getElementById("stabilized-badge"),
  datasetList: document.getElementById("dataset-list"),
  btnAddLayer: document.getElementById("btn-add-layer"),
  btnRemoveLayer: document.getElementById("btn-remove-layer"),
  hiddenLayersLabel: document.getElementById("hidden-layers-label"),
  networkStage: document.getElementById("network-stage"),
  networkSvg: document.getElementById("network-svg"),
  networkColumns: document.getElementById("network-columns"),
  mainHeatmap: document.getElementById("main-heatmap"),
  testLoss: document.getElementById("test-loss"),
  trainLoss: document.getElementById("train-loss"),
  lossChartCanvas: document.getElementById("loss-chart"),
  hoverBadge: document.getElementById("hover-badge"),
  popover: document.getElementById("inspector-popover"),
  popoverTitle: document.getElementById("popover-title"),
  popoverInput: document.getElementById("popover-input"),
  popoverApply: document.getElementById("popover-apply"),
};

const lossChart = new LossChart(el.lossChartCanvas);

function getActiveFeatureIds() {
  return Object.keys(FEATURES).filter((id) => state.activeFeatures.has(id));
}

function constructInput(x, y) {
  const ids = getActiveFeatureIds();
  return ids.map((id) => FEATURES[id].f(x, y));
}

function formatEpoch(n) {
  return n.toString().padStart(6, "0").replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function setPlaying(playing) {
  state.isPlaying = playing;
  el.btnStop.disabled = !playing;
  updateLinkStyles();
}

function startTraining() {
  resetNetwork();
  setPlaying(true);
}

function stopTraining() {
  setPlaying(false);
}

function regenerateData() {
  const { trainData, testData } = generateData(
    state.datasetKey,
    state.problemType,
    state.noise,
    state.trainRatio,
    400
  );
  state.trainData = trainData;
  state.testData = testData;
}

function resetNetwork() {
  setPlaying(false);
  state.epoch = 0;
  state.lowLossStreak = 0;
  state.stabilizedEpoch = null;
  el.epochCounter.textContent = formatEpoch(0);
  el.stabilizedBadge.hidden = true;
  el.stabilizedBadge.textContent = "";
  lossChart.reset();

  const inputIds = getActiveFeatureIds();
  const shape = [inputIds.length, ...state.hiddenLayers, 1];
  const activation = Activations[state.activationKey];
  const outputActivation =
    state.problemType === "regression" ? Activations.linear : Activations.tanh;
  const regularization = Regularization[state.regKey];

  state.network = buildNetwork(
    shape,
    activation,
    outputActivation,
    regularization,
    inputIds
  );

  renderNetworkDOM();
  updateUI();
}

function stepEpoch() {
  const data = state.trainData;
  shuffle(data);

  for (let i = 0; i < data.length; i++) {
    const p = data[i];
    forwardProp(state.network, constructInput(p.x, p.y));
    backProp(state.network, p.label);
    if ((i + 1) % state.batchSize === 0) {
      updateWeights(state.network, state.learningRate, state.regRate);
    }
  }
  // Flush remaining gradients if data.length is not divisible by batchSize
  if (data.length % state.batchSize !== 0) {
    updateWeights(state.network, state.learningRate, state.regRate);
  }

  state.epoch++;
  el.epochCounter.textContent = formatEpoch(state.epoch);
  updateUI();
}

function updateUI() {
  if (!state.network) return;
  const trainL = computeLoss(state.network, state.trainData, constructInput);
  const testL = computeLoss(state.network, state.testData, constructInput);

  el.trainLoss.textContent = trainL.toFixed(3);
  el.testLoss.textContent = testL.toFixed(3);

  if (state.epoch > 0) {
    lossChart.addPoint(trainL, testL);

    if (state.stabilizedEpoch === null) {
      if (testL < 0.005) {
        state.lowLossStreak++;
        if (state.lowLossStreak >= 5) {
          state.stabilizedEpoch = state.epoch - 4;
          el.stabilizedBadge.textContent = `Test loss < 0.005 @ ${formatEpoch(state.stabilizedEpoch)}`;
          el.stabilizedBadge.hidden = false;
        }
      } else {
        state.lowLossStreak = 0;
      }
    }
  }

  updateHeatmaps();
  updateLinkStyles();
  updateBiasTags();
}

function updateHeatmaps() {
  // Compute grid for every node in a single pass per grid point
  const activeIds = getActiveFeatureIds();
  const nodeGridsMain = new Map();
  const nodeGridsMini = new Map();

  // Main grid (only need hoveredNodeId or output node)
  const outputNode = state.network[state.network.length - 1][0];
  const targetMainId = state.hoveredNodeId || outputNode.id;

  const mainValues = evaluateGrid(GRID_MAIN, (x, y) => {
    const inputs = activeIds.map((id) => FEATURES[id].f(x, y));
    forwardProp(state.network, inputs);
    let val = outputNode.output;
    if (state.hoveredNodeId) {
      forEachNode(state.network, true, (n) => {
        if (n.id === targetMainId) val = n.output;
      });
    }
    return val;
  });

  renderHeatmap(el.mainHeatmap, mainValues, GRID_MAIN, {
    discretize: state.discretize && !state.hoveredNodeId && state.problemType === "classification",
    trainData: state.trainData,
    testData: state.testData,
    showTest: state.showTest,
  });

  // Mini heatmaps for all hidden & output nodes
  forEachNode(state.network, true, (node) => {
    nodeGridsMini.set(node.id, new Float32Array(GRID_MINI * GRID_MINI));
  });

  const step = 12 / (GRID_MINI - 1);
  for (let row = 0; row < GRID_MINI; row++) {
    const y = 6 - row * step;
    for (let col = 0; col < GRID_MINI; col++) {
      const x = -6 + col * step;
      const inputs = activeIds.map((id) => FEATURES[id].f(x, y));
      forwardProp(state.network, inputs);
      const idx = row * GRID_MINI + col;
      forEachNode(state.network, true, (node) => {
        nodeGridsMini.get(node.id)[idx] = node.output;
      });
    }
  }

  forEachNode(state.network, true, (node) => {
    const canvas = state.nodeCanvases.get(node.id);
    if (canvas) {
      renderHeatmap(canvas, nodeGridsMini.get(node.id), GRID_MINI);
    }
  });
}

function updateLinkStyles() {
  if (!state.network) return;
  forEachNode(state.network, true, (node) => {
    for (const link of node.inputLinks) {
      const path = state.linkPaths.get(link.id);
      if (!path) continue;
      if (link.isDead) {
        path.style.opacity = "0.08";
        path.setAttribute("stroke", "#94a3b8");
        path.setAttribute("stroke-width", "1");
        path.classList.remove("link-animated");
        continue;
      }
      const w = link.weight;
      const absW = Math.min(4, Math.abs(w));
      const strokeWidth = Math.max(0.75, absW * 1.8);
      path.setAttribute("stroke", w >= 0 ? "#0877bd" : "#f59322");
      path.setAttribute("stroke-width", strokeWidth.toFixed(2));
      path.style.opacity = Math.min(0.95, 0.25 + absW * 0.25).toFixed(2);
      if (state.isPlaying) {
        path.classList.add("link-animated");
      } else {
        path.classList.remove("link-animated");
      }
    }
  });
}

function updateBiasTags() {
  if (!state.network) return;
  forEachNode(state.network, true, (node) => {
    const tag = state.biasTags.get(node.id);
    if (tag) {
      tag.style.backgroundColor = valueToCssColor(Math.tanh(node.bias));
      tag.title = `Bias: ${node.bias.toFixed(3)} (click to edit)`;
    }
  });
}

function renderDatasetCards() {
  el.datasetList.innerHTML = "";
  const registry =
    state.problemType === "regression"
      ? REGRESSION_DATASETS
      : CLASSIFICATION_DATASETS;

  if (!registry[state.datasetKey]) {
    state.datasetKey = Object.keys(registry)[0];
  }

  for (const [key, entry] of Object.entries(registry)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `dataset-card ${key === state.datasetKey ? "active" : ""}`;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", key === state.datasetKey ? "true" : "false");

    const canvas = document.createElement("canvas");
    canvas.width = 68;
    canvas.height = 68;
    const previewPts = entry.generator(120, 0);
    const blankGrid = new Float32Array(20 * 20);
    renderHeatmap(canvas, blankGrid, 20, { trainData: previewPts });

    const label = document.createElement("span");
    label.textContent = entry.name;

    btn.appendChild(canvas);
    btn.appendChild(label);
    btn.addEventListener("click", () => {
      state.datasetKey = key;
      renderDatasetCards();
      regenerateData();
      resetNetwork();
    });
    el.datasetList.appendChild(btn);
  }
}

function renderNetworkDOM() {
  el.networkColumns.innerHTML = "";
  el.networkSvg.innerHTML = "";
  state.nodeCanvases.clear();
  state.linkPaths.clear();
  state.biasTags.clear();
  hidePopover();

  const numHidden = state.hiddenLayers.length;
  el.hiddenLayersLabel.textContent = `${numHidden} hidden layer${numHidden === 1 ? "" : "s"}`;
  el.btnRemoveLayer.disabled = numHidden === 0;
  el.btnAddLayer.disabled = numHidden >= 6;

  // 1. Input Features Column
  const inputCol = document.createElement("div");
  inputCol.className = "net-column";
  const inputHeader = document.createElement("div");
  inputHeader.className = "col-header";
  inputHeader.innerHTML = `<span class="col-title">Inputs</span>`;
  inputCol.appendChild(inputHeader);

  const featureAnchors = new Map();

  for (const [id, feat] of Object.entries(FEATURES)) {
    const isActive = state.activeFeatures.has(id);
    const pill = document.createElement("div");
    pill.className = `feature-node ${isActive ? "active" : ""}`;
    pill.title = `Toggle ${feat.label}`;

    const canvas = document.createElement("canvas");
    canvas.width = 30;
    canvas.height = 30;
    const grid = evaluateGrid(GRID_MINI, (x, y) => Math.tanh(feat.f(x, y) * 0.5));
    renderHeatmap(canvas, grid, GRID_MINI);

    const span = document.createElement("span");
    span.textContent = feat.label;

    pill.appendChild(canvas);
    pill.appendChild(span);

    pill.addEventListener("click", () => {
      if (state.activeFeatures.has(id)) {
        if (state.activeFeatures.size === 1) return; // keep at least 1 input
        state.activeFeatures.delete(id);
      } else {
        state.activeFeatures.add(id);
      }
      resetNetwork();
    });

    inputCol.appendChild(pill);
    if (isActive) {
      featureAnchors.set(id, pill);
    }
  }
  el.networkColumns.appendChild(inputCol);

  const nodeElements = new Map();
  for (const [id, elem] of featureAnchors.entries()) {
    nodeElements.set(id, elem);
  }

  // 2. Hidden Layer Columns
  state.hiddenLayers.forEach((count, layerIdx) => {
    const col = document.createElement("div");
    col.className = "net-column";

    const header = document.createElement("div");
    header.className = "col-header";
    const btns = document.createElement("div");
    btns.className = "col-header-btns";

    const minusBtn = document.createElement("button");
    minusBtn.className = "circle-btn";
    minusBtn.textContent = "−";
    minusBtn.disabled = count <= 1;
    minusBtn.title = "Remove neuron";
    minusBtn.addEventListener("click", () => {
      if (state.hiddenLayers[layerIdx] > 1) {
        state.hiddenLayers[layerIdx]--;
        resetNetwork();
      }
    });

    const plusBtn = document.createElement("button");
    plusBtn.className = "circle-btn";
    plusBtn.textContent = "+";
    plusBtn.disabled = count >= 8;
    plusBtn.title = "Add neuron";
    plusBtn.addEventListener("click", () => {
      if (state.hiddenLayers[layerIdx] < 8) {
        state.hiddenLayers[layerIdx]++;
        resetNetwork();
      }
    });

    btns.appendChild(minusBtn);
    btns.appendChild(plusBtn);

    const title = document.createElement("span");
    title.className = "col-title";
    title.textContent = `${count} neuron${count > 1 ? "s" : ""}`;

    header.appendChild(btns);
    header.appendChild(title);
    col.appendChild(header);

    const layerNodes = state.network[layerIdx + 1];
    for (const node of layerNodes) {
      const wrap = createNeuronElement(node);
      col.appendChild(wrap);
      nodeElements.set(node.id, wrap.querySelector(".neuron-card"));
    }

    el.networkColumns.appendChild(col);
  });

  // 3. Output Column
  const outCol = document.createElement("div");
  outCol.className = "net-column";
  const outHeader = document.createElement("div");
  outHeader.className = "col-header";
  outHeader.innerHTML = `<span class="col-title">Output</span>`;
  outCol.appendChild(outHeader);

  const outputNode = state.network[state.network.length - 1][0];
  const outWrap = createNeuronElement(outputNode);
  outCol.appendChild(outWrap);
  nodeElements.set(outputNode.id, outWrap.querySelector(".neuron-card"));
  el.networkColumns.appendChild(outCol);

  // Draw SVG Bezier links after layout
  requestAnimationFrame(() => drawLinks(nodeElements));
}

function createNeuronElement(node) {
  const wrap = document.createElement("div");
  wrap.className = "neuron-wrapper";

  const card = document.createElement("div");
  card.className = "neuron-card";
  const canvas = document.createElement("canvas");
  canvas.width = 38;
  canvas.height = 38;
  card.appendChild(canvas);

  card.addEventListener("mouseenter", () => {
    state.hoveredNodeId = node.id;
    el.hoverBadge.hidden = false;
    updateHeatmaps();
  });
  card.addEventListener("mouseleave", () => {
    state.hoveredNodeId = null;
    el.hoverBadge.hidden = true;
    updateHeatmaps();
  });

  const biasTag = document.createElement("div");
  biasTag.className = "bias-tag";
  biasTag.addEventListener("click", (e) => {
    e.stopPropagation();
    openPopover("Bias", node.bias, e.currentTarget, { type: "bias", ref: node });
  });

  wrap.appendChild(card);
  wrap.appendChild(biasTag);

  state.nodeCanvases.set(node.id, canvas);
  state.biasTags.set(node.id, biasTag);
  return wrap;
}

function drawLinks(nodeElements) {
  el.networkSvg.innerHTML = "";
  state.linkPaths.clear();
  const stageRect = el.networkStage.getBoundingClientRect();

  forEachNode(state.network, true, (node) => {
    const destEl = nodeElements.get(node.id);
    if (!destEl) return;
    const destRect = destEl.getBoundingClientRect();
    const x2 = destRect.left - stageRect.left;
    const y2 = destRect.top + destRect.height / 2 - stageRect.top;

    for (const link of node.inputLinks) {
      const srcEl = nodeElements.get(link.source.id);
      if (!srcEl) continue;
      const srcRect = srcEl.getBoundingClientRect();
      const x1 = srcRect.right - stageRect.left;
      const y1 = srcRect.top + srcRect.height / 2 - stageRect.top;

      const dx = (x2 - x1) * 0.5;
      const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("class", "link-path");
      path.addEventListener("click", (e) => {
        e.stopPropagation();
        openPopover("Weight", link.weight, e, { type: "link", ref: link });
      });

      el.networkSvg.appendChild(path);
      state.linkPaths.set(link.id, path);
    }
  });

  updateLinkStyles();
}

function openPopover(title, value, eventOrEl, target) {
  state.inspectorTarget = target;
  el.popoverTitle.textContent = title;
  el.popoverInput.value = value.toFixed(4);
  el.popover.hidden = false;

  const panelRect = el.networkStage.parentElement.getBoundingClientRect();
  let left = 20;
  let top = 20;
  if (eventOrEl instanceof MouseEvent) {
    left = eventOrEl.clientX - panelRect.left + 10;
    top = eventOrEl.clientY - panelRect.top - 10;
  } else {
    const r = eventOrEl.getBoundingClientRect();
    left = r.left - panelRect.left + 20;
    top = r.top - panelRect.top - 10;
  }
  el.popover.style.left = `${Math.max(8, left)}px`;
  el.popover.style.top = `${Math.max(8, top)}px`;
  el.popoverInput.focus();
  el.popoverInput.select();
}

function hidePopover() {
  el.popover.hidden = true;
  state.inspectorTarget = null;
}

// Animation Loop
function loop() {
  if (state.isPlaying) {
    stepEpoch();
  }
  requestAnimationFrame(loop);
}

// Wire up controls
el.btnStart.addEventListener("click", startTraining);
el.btnStop.addEventListener("click", stopTraining);

el.btnAddLayer.addEventListener("click", () => {
  if (state.hiddenLayers.length < 6) {
    state.hiddenLayers.push(2);
    resetNetwork();
  }
});
el.btnRemoveLayer.addEventListener("click", () => {
  if (state.hiddenLayers.length > 0) {
    state.hiddenLayers.pop();
    resetNetwork();
  }
});

el.popoverApply.addEventListener("click", () => {
  if (!state.inspectorTarget) return;
  const val = parseFloat(el.popoverInput.value);
  if (!Number.isNaN(val)) {
    if (state.inspectorTarget.type === "link") {
      state.inspectorTarget.ref.weight = val;
      state.inspectorTarget.ref.isDead = false;
    } else if (state.inspectorTarget.type === "bias") {
      state.inspectorTarget.ref.bias = val;
    }
    updateUI();
  }
  hidePopover();
});

document.addEventListener("click", (e) => {
  if (!el.popover.hidden && !el.popover.contains(e.target)) {
    hidePopover();
  }
});

window.addEventListener("resize", () => {
  if (state.network) {
    const nodeElements = new Map();
    document.querySelectorAll(".feature-node.active").forEach((elem, i) => {
      const id = getActiveFeatureIds()[i];
      if (id) nodeElements.set(id, elem);
    });
    // Re-render DOM links cleanly on resize
    renderNetworkDOM();
    updateUI();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") {
    e.preventDefault();
    if (state.isPlaying) {
      stopTraining();
    } else {
      startTraining();
    }
  }
});

// Initialize
renderDatasetCards();
regenerateData();
resetNetwork();
requestAnimationFrame(loop);
