// Update the testing percentage whenever the user changes training percentage.
document.addEventListener("DOMContentLoaded", function () {
    const trainingInput = document.getElementById("train-percent");
    const testingInput = document.getElementById("test-percent");
    const decisionTreeCheckbox = document.getElementById(
        "select-decision-tree",
    );
    const kMeansCheckbox = document.getElementById(
        "select-kmeans",
    );
    const decisionTreeSettings = document.getElementById(
        "decision-tree-settings",
    );
    const decisionTreeSplitSettings = document.querySelectorAll(
        ".decision-tree-split-setting",
    );
    const kMeansSettings = document.getElementById(
        "kmeans-settings",
    );

    // These inputs only exist after a dataset has been rendered.
    if (!trainingInput || !testingInput) {
        return;
    }

    function updateTestingPercentage() {
        const trainingPercentage = Number(trainingInput.value);

        if (!Number.isFinite(trainingPercentage)) {
            testingInput.value = "";
            return;
        }

        testingInput.value = 100 - trainingPercentage;
    }

    trainingInput.addEventListener(
        "input",
        updateTestingPercentage,
    );

    updateTestingPercentage();

    function updateVisibleModelSettings() {
        const decisionTreeSelected = decisionTreeCheckbox.checked;
        const kMeansSelected = kMeansCheckbox.checked;

        decisionTreeSettings.hidden = !decisionTreeSelected;
        trainingInput.disabled = !decisionTreeSelected;
        testingInput.disabled = !decisionTreeSelected;
        decisionTreeSplitSettings.forEach(function (setting) {
            setting.hidden = !decisionTreeSelected;
        });

        kMeansSettings.hidden = !kMeansSelected;
        document.getElementById("cluster-count").disabled = !kMeansSelected;
    }

    decisionTreeCheckbox.addEventListener(
        "change",
        updateVisibleModelSettings,
    );
    kMeansCheckbox.addEventListener(
        "change",
        updateVisibleModelSettings,
    );

    updateVisibleModelSettings();
});
