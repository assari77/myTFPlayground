/**
 * Lightweight SVG/Canvas sparkline chart for Training & Test Loss over epochs.
 */

export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.trainHistory = [];
    this.testHistory = [];
    this.maxPoints = 250;
  }

  reset() {
    this.trainHistory = [];
    this.testHistory = [];
    this.render();
  }

  addPoint(trainLoss, testLoss) {
    this.trainHistory.push(trainLoss);
    this.testHistory.push(testLoss);
    if (this.trainHistory.length > this.maxPoints) {
      this.trainHistory.shift();
      this.testHistory.shift();
    }
    this.render();
  }

  render() {
    const { canvas, ctx, trainHistory, testHistory } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Subtle baseline grid
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h - 1);
    ctx.lineTo(w, h - 1);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (trainHistory.length < 2) return;

    let maxVal = 0.01;
    for (let i = 0; i < trainHistory.length; i++) {
      if (trainHistory[i] > maxVal) maxVal = trainHistory[i];
      if (testHistory[i] > maxVal) maxVal = testHistory[i];
    }
    maxVal *= 1.08;

    const drawSeries = (data, color, dash = []) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.75;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * (w - 4) + 2;
        const y = h - 3 - (Math.min(maxVal, data[i]) / maxVal) * (h - 6);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    // Test loss (solid dark slate) and Training loss (lighter muted blue-gray)
    drawSeries(trainHistory, "rgba(100, 116, 139, 0.55)");
    drawSeries(testHistory, "#1e293b");
  }
}
