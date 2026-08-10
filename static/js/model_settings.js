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
    const kMeansSettings = document.getElementById("kmeans-settings");

    const uploadTrainInput = document.getElementById(
        "upload-train-percent",
    );
    const uploadModelFields = document.getElementById(
        "upload-model-fields",
    );

    const viewLinks = document.querySelectorAll("[data-view-link]");
    const workspaceViews = document.querySelectorAll(".workspace-view");

    // ----------------------------------------------------
    // Comparison and Model Library Navigation
    // ----------------------------------------------------

    function showWorkspace(viewName) {
        workspaceViews.forEach(function (view) {
            view.hidden = view.id !== viewName;
        });

        viewLinks.forEach(function (link) {
            if (link.dataset.viewLink === viewName) {
                link.setAttribute("aria-current", "page");
            } else {
                link.removeAttribute("aria-current");
            }
        });

        window.history.replaceState(
            null,
            "",
            `#${viewName}`,
        );
    }

    viewLinks.forEach(function (link) {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            showWorkspace(link.dataset.viewLink);
        });
    });

    const firstView = (
        window.location.hash === "#model-library"
            ? "model-library"
            : "comparison"
    );
    showWorkspace(firstView);

    // ----------------------------------------------------
    // Remember Selected Models
    // ----------------------------------------------------

    const modelCheckboxes = [
        decisionTreeCheckbox,
        kMeansCheckbox,
    ].filter(Boolean);

    try {
        const rememberedModels = JSON.parse(
            window.sessionStorage.getItem("selected_models"),
        );

        if (Array.isArray(rememberedModels)) {
            modelCheckboxes.forEach(function (checkbox) {
                checkbox.checked = rememberedModels.includes(
                    checkbox.value,
                );
            });
        }
    } catch (_error) {
        // Keep the model choices provided by Flask when storage is unavailable.
    }

    function selectedModelNames() {
        return modelCheckboxes
            .filter(function (checkbox) {
                return checkbox.checked;
            })
            .map(function (checkbox) {
                return checkbox.value;
            });
    }

    // Add the remembered model choices to the dataset rendering form.
    function writeModelFields(container, selectedModels) {
        if (!container) {
            return;
        }

        container.replaceChildren();

        selectedModels.forEach(function (modelName) {
            const hiddenInput = document.createElement("input");
            hiddenInput.type = "hidden";
            hiddenInput.name = "selected_models";
            hiddenInput.value = modelName;
            container.appendChild(hiddenInput);
        });
    }

    function updateSelectedModels() {
        const selectedModels = selectedModelNames();

        try {
            window.sessionStorage.setItem(
                "selected_models",
                JSON.stringify(selectedModels),
            );
        } catch (_error) {
            // The current page still remembers choices without browser storage.
        }

        writeModelFields(uploadModelFields, selectedModels);

        // The green panel only displays parameters for selected models.
        function updateParameterSection(section, isSelected) {
            if (!section) {
                return;
            }

            section.hidden = !isSelected;
            section.querySelectorAll("input, select").forEach(
                function (control) {
                    control.disabled = !isSelected;
                },
            );
        }

        if (decisionTreeCheckbox) {
            updateParameterSection(
                decisionTreeSettings,
                decisionTreeCheckbox.checked,
            );
        }

        if (kMeansCheckbox) {
            updateParameterSection(
                kMeansSettings,
                kMeansCheckbox.checked,
            );
        }
    }

    modelCheckboxes.forEach(function (checkbox) {
        checkbox.addEventListener(
            "change",
            updateSelectedModels,
        );
    });

    updateSelectedModels();

    // ----------------------------------------------------
    // Linked Train and Test Percentages
    // ----------------------------------------------------

    function updateTestingPercentage() {
        const trainingPercentage = Number(trainingInput.value);

        if (!Number.isFinite(trainingPercentage)) {
            testingInput.value = "";
            return;
        }

        testingInput.value = 100 - trainingPercentage;

        if (uploadTrainInput) {
            uploadTrainInput.value = trainingPercentage;
        }

    }

    if (trainingInput && testingInput) {
        trainingInput.addEventListener(
            "input",
            updateTestingPercentage,
        );

        updateTestingPercentage();
    }
});
