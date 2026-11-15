import test from "node:test";
import assert from "node:assert/strict";
import {
  Activations,
  Regularization,
  buildNetwork,
  forwardProp,
  backProp,
  updateWeights,
  computeLoss,
  createRng,
} from "../src/nn.js";
import { generateData, FEATURES } from "../src/datasets.js";

test("Neural network reduces loss on XOR classification", () => {
  const rng = createRng(42);
  const { trainData, testData } = generateData("xor", "classification", 0, 50, 200, rng);
  const inputIds = ["x", "y"];
  const constructInput = (x, y) => inputIds.map((id) => FEATURES[id].f(x, y));

  const network = buildNetwork(
    [2, 4, 2, 1],
    Activations.tanh,
    Activations.tanh,
    Regularization.none,
    inputIds,
    false,
    rng
  );

  const initialLoss = computeLoss(network, trainData, constructInput);

  for (let epoch = 0; epoch < 120; epoch++) {
    for (let i = 0; i < trainData.length; i++) {
      const p = trainData[i];
      forwardProp(network, constructInput(p.x, p.y));
      backProp(network, p.label);
      if ((i + 1) % 10 === 0) {
        updateWeights(network, 0.03, 0);
      }
    }
    updateWeights(network, 0.03, 0);
  }

  const finalTrainLoss = computeLoss(network, trainData, constructInput);
  const finalTestLoss = computeLoss(network, testData, constructInput);

  assert.ok(finalTrainLoss < initialLoss * 0.2, `Expected train loss (${finalTrainLoss}) << initial (${initialLoss})`);
  assert.ok(finalTestLoss < 0.15, `Expected test loss (${finalTestLoss}) < 0.15`);
});

test("L1 regularization zeros out small weights", () => {
  const rng = createRng(7);
  const network = buildNetwork(
    [2, 2, 1],
    Activations.linear,
    Activations.linear,
    Regularization.l1,
    ["x", "y"],
    false,
    rng
  );

  // Force a small positive weight and apply L1 update
  const link = network[1][0].inputLinks[0];
  link.weight = 0.005;
  forwardProp(network, [0, 0]);
  backProp(network, 0);
  updateWeights(network, 0.1, 0.1);

  assert.equal(link.weight, 0);
  assert.equal(link.isDead, true);
});

test("Regression dataset generates continuous labels in [-1, 1]", () => {
  const rng = createRng(123);
  const { trainData, testData } = generateData("regGauss", "regression", 5, 60, 100, rng);
  assert.equal(trainData.length, 60);
  assert.equal(testData.length, 40);
  for (const p of trainData) {
    assert.ok(p.label >= -1 && p.label <= 1);
  }
});
