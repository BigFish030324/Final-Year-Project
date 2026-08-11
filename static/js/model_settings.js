document.addEventListener("DOMContentLoaded", function () {
    const applicationLayout = document.querySelector(
        ".application-layout",
    );
    const sidebarCollapseButton = document.getElementById(
        "sidebar-collapse",
    );
    const brandLink = document.querySelector(".brand");

    const trainingInput = document.getElementById("train-percent");
    const testingInput = document.getElementById("test-percent");
    const uploadTrainInput = document.getElementById(
        "upload-train-percent",
    );
    const uploadModelFields = document.getElementById(
        "upload-model-fields",
    );

    const decisionTreeSettings = document.getElementById(
        "decision-tree-settings",
    );
    const kMeansSettings = document.getElementById("kmeans-settings");
    const modelCheckboxes = Array.from(
        document.querySelectorAll("[data-model-choice]"),
    );

    const viewLinks = document.querySelectorAll("[data-view-link]");
    const workspaceViews = document.querySelectorAll(".workspace-view");
    const modelDock = document.getElementById("model-dock");
    const modelDockToggle = document.getElementById(
        "model-dock-toggle",
    );

    // ----------------------------------------------------
    // Collapsible Left Sidebar
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

    function syncModelDockSpacing() {
        if (!modelDock || modelDock.hidden) {
            document.body.classList.remove("model-dock-visible");
            return;
        }

        document.body.classList.add("model-dock-visible");
        document.documentElement.style.setProperty(
            "--model-dock-clearance",
            `${Math.ceil(modelDock.getBoundingClientRect().height) + 34}px`,
        );
    }

    function showWorkspace(viewName) {
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

        if (modelDock) {
            modelDock.hidden = viewName !== "comparison";
        }
        syncModelDockSpacing();

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
    // Yellow Bottom Model Dock
    // ----------------------------------------------------

    function updateModelDockCollapse(collapsed) {
        modelDock.classList.toggle("collapsed", collapsed);
        modelDockToggle.setAttribute(
            "aria-expanded",
            String(!collapsed),
        );
        modelDockToggle.setAttribute(
            "aria-label",
            collapsed
                ? "Expand data model selection"
                : "Minimize data model selection",
        );
        modelDockToggle.title = (
            collapsed
                ? "Expand data model selection"
                : "Minimize data model selection"
        );
        window.requestAnimationFrame(syncModelDockSpacing);
    }

    let rememberedDockState = false;
    try {
        rememberedDockState = (
            window.localStorage.getItem("model_dock_collapsed") === "true"
        );
    } catch (_error) {
        // Use the expanded model dock when storage is unavailable.
    }
    updateModelDockCollapse(rememberedDockState);

    modelDockToggle.addEventListener("click", function () {
        const collapsed = !modelDock.classList.contains("collapsed");
        updateModelDockCollapse(collapsed);

        try {
            window.localStorage.setItem(
                "model_dock_collapsed",
                String(collapsed),
            );
        } catch (_error) {
            // The dock still works for the current page.
        }
    });

    if (window.ResizeObserver) {
        new ResizeObserver(syncModelDockSpacing).observe(modelDock);
    }
    window.addEventListener("resize", syncModelDockSpacing);

    // ----------------------------------------------------
    // Synchronized Model Choices
    // ----------------------------------------------------

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
        // Keep the model choices provided by Flask.
    }

    function selectedModelNames() {
        return Array.from(
            new Set(
                modelCheckboxes
                    .filter(function (checkbox) {
                        return checkbox.checked;
                    })
                    .map(function (checkbox) {
                        return checkbox.value;
                    }),
            ),
        );
    }

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

    function updateSelectedModels() {
        const selectedModels = selectedModelNames();

        try {
            window.sessionStorage.setItem(
                "selected_models",
                JSON.stringify(selectedModels),
            );
        } catch (_error) {
            // The current page still remembers matching checkbox states.
        }

        writeModelFields(uploadModelFields, selectedModels);
        updateParameterSection(
            decisionTreeSettings,
            selectedModels.includes("decision_tree"),
        );
        updateParameterSection(
            kMeansSettings,
            selectedModels.includes("kmeans"),
        );
    }

    modelCheckboxes.forEach(function (checkbox) {
        checkbox.addEventListener("change", function () {
            modelCheckboxes.forEach(function (matchingCheckbox) {
                if (matchingCheckbox.value === checkbox.value) {
                    matchingCheckbox.checked = checkbox.checked;
                }
            });
            updateSelectedModels();
        });
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
        trainingInput.addEventListener("input", updateTestingPercentage);
        updateTestingPercentage();
    }
});
