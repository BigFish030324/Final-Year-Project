"use strict";

// Shared Canvas Preparation
function prepareChartCanvas(canvas, chartHeight) {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const chartWidth = Math.max(canvas.clientWidth, 320);

    canvas.style.height = `${chartHeight}px`;
    canvas.width = Math.round(chartWidth * devicePixelRatio);
    canvas.height = Math.round(chartHeight * devicePixelRatio);

    const drawingContext = canvas.getContext("2d");
    drawingContext.setTransform(
        devicePixelRatio,
        0,
        0,
        devicePixelRatio,
        0,
        0,
    );
    drawingContext.clearRect(0, 0, chartWidth, chartHeight);
    drawingContext.font = "12px system-ui, sans-serif";
    drawingContext.textBaseline = "middle";

    return {
        drawingContext,
        chartWidth,
        chartHeight,
    };
}

// Shared Chart Data
function readChartData(chartDataElementId) {
    const chartDataElement = document.getElementById(chartDataElementId);

    if (!chartDataElement) {
        return null;
    }

    try {
        return JSON.parse(chartDataElement.textContent);
    } catch (_error) {
        return null;
    }
}

function formatChartLabel(label) {
    return String(label)
        .replaceAll("_", " ")
        .replace(/\b\w/g, function (letter) {
            return letter.toUpperCase();
        });
}

function formatChartNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "Unavailable";
    }

    return number.toLocaleString(undefined, {
        maximumFractionDigits: 5,
    });
}

// Shared Metric Bar Chart
function drawMetricBarChart(canvasId, metrics, barColour) {
    const canvas = document.getElementById(canvasId);
    const availableMetrics = Object.entries(metrics || {}).filter(
        function (metric) {
            return Number.isFinite(Number(metric[1]));
        },
    );

    if (!canvas || availableMetrics.length === 0) {
        return;
    }

    const chartHeight = Math.max(170, 50 + availableMetrics.length * 42);
    const preparedCanvas = prepareChartCanvas(canvas, chartHeight);
    const drawingContext = preparedCanvas.drawingContext;
    const chartWidth = preparedCanvas.chartWidth;
    const labelWidth = Math.min(145, chartWidth * 0.36);
    const valueWidth = 74;
    const barStartingPosition = labelWidth;
    const maximumBarWidth = Math.max(
        80,
        chartWidth - labelWidth - valueWidth - 18,
    );
    const maximumAbsoluteValue = Math.max(
        ...availableMetrics.map(function (metric) {
            return Math.abs(Number(metric[1]));
        }),
        1,
    );

    availableMetrics.forEach(function (metric, metricIndex) {
        const metricName = metric[0];
        const metricValue = Number(metric[1]);
        const rowPosition = 36 + metricIndex * 42;
        const barWidth = Math.max(
            2,
            Math.abs(metricValue) / maximumAbsoluteValue * maximumBarWidth,
        );

        drawingContext.fillStyle = "#1f2937";
        drawingContext.textAlign = "left";
        drawingContext.fillText(
            formatChartLabel(metricName),
            4,
            rowPosition,
            labelWidth - 10,
        );

        drawingContext.fillStyle = "#e2e8f0";
        drawingContext.fillRect(
            barStartingPosition,
            rowPosition - 10,
            maximumBarWidth,
            20,
        );

        drawingContext.fillStyle = metricValue < 0 ? "#c2413b" : barColour;
        drawingContext.fillRect(
            barStartingPosition,
            rowPosition - 10,
            barWidth,
            20,
        );

        drawingContext.fillStyle = "#1f2937";
        drawingContext.textAlign = "right";
        drawingContext.fillText(
            formatChartNumber(metricValue),
            chartWidth - 4,
            rowPosition,
        );
    });
}

// Decision Tree - Metric Chart
function drawDecisionTreeMetricChart(decisionTreeResult) {
    drawMetricBarChart(
        "decision-tree-metric-chart",
        decisionTreeResult.metrics,
        "#087aa6",
    );
}

// Decision Tree - Confusion Matrix Chart
function drawConfusionMatrixChart(decisionTreeResult) {
    const canvas = document.getElementById(
        "decision-tree-confusion-matrix-chart",
    );
    const labels = decisionTreeResult.chart.labels || [];
    const matrix = decisionTreeResult.chart.matrix || [];

    if (!canvas || labels.length === 0 || matrix.length === 0) {
        return;
    }

    const chartHeight = Math.max(300, Math.min(480, labels.length * 54 + 145));
    const preparedCanvas = prepareChartCanvas(canvas, chartHeight);
    const drawingContext = preparedCanvas.drawingContext;
    const chartWidth = preparedCanvas.chartWidth;
    const leftSpace = Math.min(115, chartWidth * 0.3);
    const topSpace = 80;
    const availableGridWidth = chartWidth - leftSpace - 24;
    const availableGridHeight = chartHeight - topSpace - 42;
    const cellSize = Math.max(
        24,
        Math.min(
            availableGridWidth / labels.length,
            availableGridHeight / labels.length,
            64,
        ),
    );
    const maximumValue = Math.max(...matrix.flat().map(Number), 1);

    drawingContext.fillStyle = "#475569";
    drawingContext.textAlign = "center";
    drawingContext.fillText(
        "Predicted Class",
        leftSpace + cellSize * labels.length / 2,
        18,
    );

    drawingContext.save();
    drawingContext.translate(18, topSpace + cellSize * labels.length / 2);
    drawingContext.rotate(-Math.PI / 2);
    drawingContext.fillText("Actual Class", 0, 0);
    drawingContext.restore();

    labels.forEach(function (label, labelIndex) {
        drawingContext.fillStyle = "#334155";
        drawingContext.textAlign = "center";
        drawingContext.fillText(
            String(label),
            leftSpace + labelIndex * cellSize + cellSize / 2,
            topSpace - 25,
            cellSize - 4,
        );

        drawingContext.textAlign = "right";
        drawingContext.fillText(
            String(label),
            leftSpace - 10,
            topSpace + labelIndex * cellSize + cellSize / 2,
            leftSpace - 40,
        );
    });

    matrix.forEach(function (matrixRow, rowIndex) {
        matrixRow.forEach(function (cellValue, columnIndex) {
            const number = Number(cellValue);
            const cellStrength = 0.15 + 0.8 * number / maximumValue;
            const cellStartingX = leftSpace + columnIndex * cellSize;
            const cellStartingY = topSpace + rowIndex * cellSize;

            drawingContext.fillStyle = `rgba(8, 122, 166, ${cellStrength})`;
            drawingContext.fillRect(
                cellStartingX + 1,
                cellStartingY + 1,
                cellSize - 2,
                cellSize - 2,
            );

            drawingContext.fillStyle = cellStrength > 0.55 ? "#ffffff" : "#172033";
            drawingContext.textAlign = "center";
            drawingContext.fillText(
                formatChartNumber(number),
                cellStartingX + cellSize / 2,
                cellStartingY + cellSize / 2,
            );
        });
    });
}

// Decision Tree - Regression Prediction Chart
function drawRegressionPredictionChart(decisionTreeResult) {
    const canvas = document.getElementById(
        "decision-tree-regression-prediction-chart",
    );
    const actualValues = (decisionTreeResult.chart.actual || []).map(Number);
    const predictedValues = (decisionTreeResult.chart.predicted || []).map(Number);

    if (!canvas || actualValues.length === 0 || predictedValues.length === 0) {
        return;
    }

    const preparedCanvas = prepareChartCanvas(canvas, 300);
    const drawingContext = preparedCanvas.drawingContext;
    const chartWidth = preparedCanvas.chartWidth;
    const chartHeight = preparedCanvas.chartHeight;
    const allValues = actualValues.concat(predictedValues).filter(Number.isFinite);
    const minimumValue = Math.min(...allValues);
    const maximumValue = Math.max(...allValues);
    const valueRange = maximumValue - minimumValue || 1;
    const chartLeft = 48;
    const chartRight = chartWidth - 20;
    const chartTop = 46;
    const chartBottom = chartHeight - 42;
    const pointCount = Math.max(actualValues.length, predictedValues.length);

    drawingContext.strokeStyle = "#cbd5e1";
    drawingContext.lineWidth = 1;
    drawingContext.beginPath();
    drawingContext.moveTo(chartLeft, chartTop);
    drawingContext.lineTo(chartLeft, chartBottom);
    drawingContext.lineTo(chartRight, chartBottom);
    drawingContext.stroke();

    drawingContext.fillStyle = "#475569";
    drawingContext.textAlign = "right";
    drawingContext.fillText(formatChartNumber(maximumValue), chartLeft - 8, chartTop);
    drawingContext.fillText(formatChartNumber(minimumValue), chartLeft - 8, chartBottom);

    function drawValueLine(values, lineColour) {
        drawingContext.strokeStyle = lineColour;
        drawingContext.fillStyle = lineColour;
        drawingContext.lineWidth = 2.5;
        drawingContext.beginPath();

        values.forEach(function (value, valueIndex) {
            const pointX = chartLeft + (
                pointCount === 1
                    ? (chartRight - chartLeft) / 2
                    : valueIndex / (pointCount - 1) * (chartRight - chartLeft)
            );
            const pointY = chartBottom - (
                (value - minimumValue) / valueRange * (chartBottom - chartTop)
            );

            if (valueIndex === 0) {
                drawingContext.moveTo(pointX, pointY);
            } else {
                drawingContext.lineTo(pointX, pointY);
            }
        });

        drawingContext.stroke();

        values.forEach(function (value, valueIndex) {
            const pointX = chartLeft + (
                pointCount === 1
                    ? (chartRight - chartLeft) / 2
                    : valueIndex / (pointCount - 1) * (chartRight - chartLeft)
            );
            const pointY = chartBottom - (
                (value - minimumValue) / valueRange * (chartBottom - chartTop)
            );
            drawingContext.beginPath();
            drawingContext.arc(pointX, pointY, 3, 0, Math.PI * 2);
            drawingContext.fill();
        });
    }

    drawValueLine(actualValues, "#087aa6");
    drawValueLine(predictedValues, "#7c3aed");

    drawingContext.textAlign = "left";
    drawingContext.fillStyle = "#087aa6";
    drawingContext.fillRect(chartLeft, 15, 16, 4);
    drawingContext.fillStyle = "#334155";
    drawingContext.fillText("Actual", chartLeft + 22, 17);
    drawingContext.fillStyle = "#7c3aed";
    drawingContext.fillRect(chartLeft + 80, 15, 16, 4);
    drawingContext.fillStyle = "#334155";
    drawingContext.fillText("Predicted", chartLeft + 102, 17);
}

// K-Means - Metric Chart
function drawKmeansMetricChart(kmeansResult) {
    drawMetricBarChart(
        "kmeans-metric-chart",
        kmeansResult.metrics,
        "#2f6e4f",
    );
}

// K-Means - Cluster Size Chart
function drawKmeansClusterSizeChart(kmeansResult) {
    const canvas = document.getElementById("kmeans-cluster-size-chart");
    const clusterLabels = kmeansResult.chart.labels || [];
    const clusterValues = (kmeansResult.chart.values || []).map(Number);

    if (!canvas || clusterLabels.length === 0 || clusterValues.length === 0) {
        return;
    }

    const chartHeight = Math.max(190, 60 + clusterLabels.length * 48);
    const preparedCanvas = prepareChartCanvas(canvas, chartHeight);
    const drawingContext = preparedCanvas.drawingContext;
    const chartWidth = preparedCanvas.chartWidth;
    const labelWidth = Math.min(115, chartWidth * 0.3);
    const maximumBarWidth = Math.max(100, chartWidth - labelWidth - 80);
    const maximumClusterSize = Math.max(...clusterValues, 1);

    clusterLabels.forEach(function (clusterLabel, clusterIndex) {
        const clusterSize = clusterValues[clusterIndex];
        const rowPosition = 38 + clusterIndex * 48;
        const clusterBarWidth = clusterSize / maximumClusterSize * maximumBarWidth;

        drawingContext.fillStyle = "#334155";
        drawingContext.textAlign = "left";
        drawingContext.fillText(clusterLabel, 4, rowPosition);

        drawingContext.fillStyle = "#e2e8f0";
        drawingContext.fillRect(labelWidth, rowPosition - 12, maximumBarWidth, 24);
        drawingContext.fillStyle = "#2f6e4f";
        drawingContext.fillRect(labelWidth, rowPosition - 12, clusterBarWidth, 24);

        drawingContext.fillStyle = "#334155";
        drawingContext.textAlign = "right";
        drawingContext.fillText(
            `${formatChartNumber(clusterSize)} rows`,
            chartWidth - 4,
            rowPosition,
        );
    });
}

// Draw Available Result Charts
function drawAvailableResultCharts() {
    const decisionTreeResult = readChartData("decision-tree-chart-data");
    const kmeansResult = readChartData("kmeans-chart-data");

    if (decisionTreeResult) {
        drawDecisionTreeMetricChart(decisionTreeResult);

        if (decisionTreeResult.chart.type === "confusion_matrix") {
            drawConfusionMatrixChart(decisionTreeResult);
        }

        if (decisionTreeResult.chart.type === "actual_and_predicted") {
            drawRegressionPredictionChart(decisionTreeResult);
        }
    }

    if (kmeansResult) {
        drawKmeansMetricChart(kmeansResult);
        drawKmeansClusterSizeChart(kmeansResult);
    }
}

drawAvailableResultCharts();

let resultChartResizeTimer;
window.addEventListener("resize", function () {
    window.clearTimeout(resultChartResizeTimer);
    resultChartResizeTimer = window.setTimeout(
        drawAvailableResultCharts,
        150,
    );
});
