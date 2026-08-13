document.addEventListener("DOMContentLoaded", function () {
    const applicationLayout = document.querySelector(
        ".application-layout",
    );
    const sidebarCollapseButton = document.getElementById(
        "sidebar-collapse",
    );
    const brandLink = document.querySelector(".brand");

    const trainingPercentageInput = document.getElementById(
        "training-percentage",
    );
    const testingPercentageText = document.getElementById(
        "testing-percentage-text",
    );
    const submittedTrainingPercentageInput = document.getElementById(
        "submitted-training-percentage",
    );
    const submittedDataModelFields = document.getElementById(
        "submitted-data-model-fields",
    );

    const decisionTreeParameters = document.getElementById(
        "decision-tree-parameters",
    );
    const kmeansParameters = document.getElementById("kmeans-parameters");
    const dataModelSelectionCheckboxes = Array.from(
        document.querySelectorAll("[data-model-choice]"),
    );
    const comparisonDataModelCards = Array.from(
        document.querySelectorAll("[data-comparison-model]"),
    );
    const comparisonDataModelSelectionMessage = document.getElementById(
        "comparison-data-model-selection-message",
    );
    const modelLibraryDataModelCards = Array.from(
        document.querySelectorAll("[data-model-library-card]"),
    );
    const addDataModelToComparisonButtons = document.querySelectorAll(
        "[data-add-model-to-comparison]",
    );
    let activeDataModelName = null;

    const viewLinks = document.querySelectorAll("[data-view-link]");
    const workspaceViews = document.querySelectorAll(".workspace-view");
    const dataModelSelection = document.getElementById(
        "data-model-selection",
    );
    const dataModelSelectionMinimizeButton = document.getElementById(
        "data-model-selection-minimize-button",
    );

    // ----------------------------------------------------
    // Collapsible Navigation Sidebar
    // ----------------------------------------------------

    function updateSidebarCollapse(collapsed) {
        applicationLayout.classList.toggle(
            "sidebar-collapsed",
            collapsed,
        );
        document.documentElement.classList.toggle(
            "sidebar-precollapsed",
            collapsed,
        );

        sidebarCollapseButton.setAttribute(
            "aria-label",
            collapsed ? "Expand menu" : "Minimize menu",
        );
        sidebarCollapseButton.title = (
            collapsed ? "Expand menu" : "Minimize menu"
        );
    }

    let rememberedSidebarState = false;
    try {
        rememberedSidebarState = (
            window.localStorage.getItem("sidebar_collapsed") === "true"
        );
    } catch (_error) {
        // Use the expanded sidebar when browser storage is unavailable.
    }
    updateSidebarCollapse(rememberedSidebarState);

    sidebarCollapseButton.addEventListener("click", function () {
        const collapsed = !applicationLayout.classList.contains(
            "sidebar-collapsed",
        );
        updateSidebarCollapse(collapsed);

        try {
            window.localStorage.setItem(
                "sidebar_collapsed",
                String(collapsed),
            );
        } catch (_error) {
            // The sidebar still works for the current page.
        }
    });

    brandLink.addEventListener("click", function (event) {
        event.preventDefault();

        if (applicationLayout.classList.contains("sidebar-collapsed")) {
            updateSidebarCollapse(false);

            try {
                window.localStorage.setItem("sidebar_collapsed", "false");
            } catch (_error) {
                // The expanded state still applies to the current page.
            }
            return;
        }

        showWorkspace("comparison");
    });

    // ----------------------------------------------------
    // Comparison and Model Library Navigation
    // ----------------------------------------------------

    function updateDataModelSelectionSpacing() {
        if (!dataModelSelection || dataModelSelection.hidden) {
            document.body.classList.remove("data-model-selection-visible");
            return;
        }

        document.body.classList.add("data-model-selection-visible");
        document.documentElement.style.setProperty(
            "--data-model-selection-clearance",
            `${Math.ceil(dataModelSelection.getBoundingClientRect().height) + 34}px`,
        );
    }

    function showWorkspace(viewName) {
        applicationLayout.classList.toggle(
            "model-library-open",
            viewName === "model-library",
        );

        workspaceViews.forEach(function (view) {
            view.hidden = view.id !== viewName;
        });

        viewLinks.forEach(function (link) {
            if (link.dataset.viewLink === viewName) {
                link.setAttribute("aria-current", "page");
                link.classList.add("active");
            } else {
                link.removeAttribute("aria-current");
                link.classList.remove("active");
            }
        });

        if (dataModelSelection) {
            dataModelSelection.hidden = viewName !== "comparison";
        }
        updateDataModelSelectionSpacing();

        window.history.replaceState(null, "", `#${viewName}`);
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
    // Comparison - Data Model Selection
    // ----------------------------------------------------

    function updateDataModelSelectionMinimizedState(minimized) {
        dataModelSelection.classList.toggle("collapsed", minimized);
        dataModelSelectionMinimizeButton.setAttribute(
            "aria-expanded",
            String(!minimized),
        );
        dataModelSelectionMinimizeButton.setAttribute(
            "aria-label",
            minimized
                ? "Expand data model selection"
                : "Minimize data model selection",
        );
        dataModelSelectionMinimizeButton.title = (
            minimized
                ? "Expand data model selection"
                : "Minimize data model selection"
        );
        window.requestAnimationFrame(updateDataModelSelectionSpacing);
    }

    // Begin every new page load with Data Model Selection minimized.
    updateDataModelSelectionMinimizedState(true);

    dataModelSelectionMinimizeButton.addEventListener("click", function () {
        const minimized = !dataModelSelection.classList.contains("collapsed");
        updateDataModelSelectionMinimizedState(minimized);
    });

    if (window.ResizeObserver) {
        new ResizeObserver(updateDataModelSelectionSpacing).observe(
            dataModelSelection,
        );
    }
    window.addEventListener("resize", updateDataModelSelectionSpacing);

    // ----------------------------------------------------
    // Comparison and Model Library - Synchronized Data Model Checkboxes
    // ----------------------------------------------------

    try {
        const rememberedModels = JSON.parse(
            window.sessionStorage.getItem("selected_data_models"),
        );

        if (Array.isArray(rememberedModels)) {
            dataModelSelectionCheckboxes.forEach(function (checkbox) {
                checkbox.checked = rememberedModels.includes(
                    checkbox.value,
                );
            });
        }
    } catch (_error) {
        // Keep the model choices provided by Flask.
    }

    function getSelectedDataModelNames() {
        return Array.from(
            new Set(
                dataModelSelectionCheckboxes
                    .filter(function (checkbox) {
                        return checkbox.checked;
                    })
                    .map(function (checkbox) {
                        return checkbox.value;
                    }),
            ),
        );
    }

    function writeSubmittedDataModelFields(container, selectedDataModels) {
        if (!container) {
            return;
        }

        container.replaceChildren();

        selectedDataModels.forEach(function (dataModelName) {
            const hiddenInput = document.createElement("input");
            hiddenInput.type = "hidden";
            hiddenInput.name = "selected_data_models";
            hiddenInput.value = dataModelName;
            container.appendChild(hiddenInput);
        });
    }

    function updateDataModelParameterSection(section, isSelected) {
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

    function updateSelectedDataModels() {
        const selectedDataModels = getSelectedDataModelNames();

        if (!selectedDataModels.includes(activeDataModelName)) {
            activeDataModelName = selectedDataModels[0] || null;
        }

        try {
            window.sessionStorage.setItem(
                "selected_data_models",
                JSON.stringify(selectedDataModels),
            );
        } catch (_error) {
            // The current page still remembers matching checkbox states.
        }

        writeSubmittedDataModelFields(
            submittedDataModelFields,
            selectedDataModels,
        );

        comparisonDataModelCards.forEach(function (dataModelCard) {
            const isSelected = selectedDataModels.includes(
                dataModelCard.dataset.comparisonModel,
            );
            dataModelCard.classList.toggle("selected", isSelected);
            dataModelCard.classList.toggle(
                "active",
                isSelected && dataModelCard.dataset.comparisonModel === activeDataModelName,
            );
            dataModelCard.setAttribute("aria-selected", String(isSelected));
        });

        modelLibraryDataModelCards.forEach(function (dataModelCard) {
            dataModelCard.classList.toggle(
                "selected",
                selectedDataModels.includes(dataModelCard.dataset.modelLibraryName),
            );
        });

        if (comparisonDataModelSelectionMessage) {
            if (selectedDataModels.length === 0) {
                comparisonDataModelSelectionMessage.textContent = (
                    "Choose at least one model from Data Model Selection or Model Library."
                );
            } else if (selectedDataModels.length === 1) {
                comparisonDataModelSelectionMessage.textContent = (
                    "One model is selected. The second comparison card is optional."
                );
            } else {
                comparisonDataModelSelectionMessage.textContent = (
                    "Both models are selected and ready for side-by-side comparison."
                );
            }
        }

        // Parameters - Show Only the Active Selected Data Model
        updateDataModelParameterSection(
            decisionTreeParameters,
            activeDataModelName === "decision_tree",
        );
        updateDataModelParameterSection(
            kmeansParameters,
            activeDataModelName === "kmeans",
        );
    }

    dataModelSelectionCheckboxes.forEach(function (checkbox) {
        checkbox.addEventListener("change", function () {
            dataModelSelectionCheckboxes.forEach(function (matchingCheckbox) {
                if (matchingCheckbox.value === checkbox.value) {
                    matchingCheckbox.checked = checkbox.checked;
                }
            });
            updateSelectedDataModels();
        });
    });

    function activateComparisonDataModelCard(dataModelCard) {
        const dataModelName = dataModelCard.dataset.comparisonModel;

        if (!getSelectedDataModelNames().includes(dataModelName)) {
            return;
        }

        activeDataModelName = dataModelName;
        updateSelectedDataModels();
    }

    comparisonDataModelCards.forEach(function (dataModelCard) {
        dataModelCard.addEventListener("click", function () {
            activateComparisonDataModelCard(dataModelCard);
        });

        dataModelCard.addEventListener("keydown", function (event) {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                activateComparisonDataModelCard(dataModelCard);
            }
        });
    });

    // ----------------------------------------------------
    // Model Library - Add Data Model to Comparison Button
    // ----------------------------------------------------

    addDataModelToComparisonButtons.forEach(function (button) {
        button.addEventListener("click", function () {
            const dataModelName = button.dataset.addModelToComparison;

            dataModelSelectionCheckboxes.forEach(function (checkbox) {
                if (checkbox.value === dataModelName) {
                    checkbox.checked = true;
                }
            });

            activeDataModelName = dataModelName;
            updateSelectedDataModels();
            showWorkspace("comparison");
        });
    });

    updateSelectedDataModels();

    // ----------------------------------------------------
    // Comparison - Linked Training and Testing Percentages
    // ----------------------------------------------------

    function updateTestingPercentage() {
        const trainingPercentage = Number(trainingPercentageInput.value);

        if (!Number.isFinite(trainingPercentage)) {
            testingPercentageText.value = "";
            return;
        }

        testingPercentageText.value = 100 - trainingPercentage;

        if (submittedTrainingPercentageInput) {
            submittedTrainingPercentageInput.value = trainingPercentage;
        }
    }

    if (trainingPercentageInput && testingPercentageText) {
        trainingPercentageInput.addEventListener("input", updateTestingPercentage);
        updateTestingPercentage();
    }
});
