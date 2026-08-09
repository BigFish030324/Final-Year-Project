// Update the testing percentage whenever the user changes training percentage.
document.addEventListener("DOMContentLoaded", function () {
    const trainingInput = document.getElementById("train-percent");
    const testingInput = document.getElementById("test-percent");

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
});
