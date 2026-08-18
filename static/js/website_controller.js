// --------------------
// Website Controller
// Connect every webpage control to the Flask API without using a frontend framework.
// --------------------
(() => {
    "use strict";

    const page = document.body.dataset.page;
    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
    const state = {
        session: null,
        dataset: null,
        models: [],
        slots: { a: null, b: null },
        activeSlot: null,
        result: null,
        processingMode: "economy",
        settings: { train_pct: 70, random_seed: 42, processing: "ask", tooltips: true, export: "pdf", theme: "light" },
        lastLogs: 0,
    };
    let splitInputTimer = null;

    const escapeHtml = (value) => String(value ?? "")
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

    async function api(url, options = {}) {
        const config = { ...options, headers: { ...(options.headers || {}) } };
        if (config.body && !(config.body instanceof FormData) && typeof config.body !== "string") {
            config.headers["Content-Type"] = "application/json";
            config.body = JSON.stringify(config.body);
        }
        const response = await fetch(url, config);
        let payload;
        try { payload = await response.json(); } catch { payload = { ok: false, error: `Request failed (${response.status}).` }; }
        if (!response.ok || payload.ok === false) {
            const error = new Error(payload.error || (payload.errors || []).join("\n") || "The request could not be completed.");
            error.payload = payload;
            throw error;
        }
        return payload;
    }

    function toast(message, type = "success", duration = 4300) {
        const region = $("#toast-region");
        if (!region) return;
        const item = document.createElement("div");
        item.className = `toast ${type}`;
        item.textContent = message;
        region.appendChild(item);
        setTimeout(() => item.remove(), duration);
    }

    function openModal(html, options = {}) {
        const modal = $("#global-modal");
        $("#modal-content").innerHTML = html;
        modal.classList.toggle("no-close", options.hideClose === true);
        modal.classList.remove("hidden");
    }

    function closeModal() {
        $("#global-modal")?.classList.add("hidden");
    }

    function formatNumber(value, digits = 4) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
        const number = Number(value);
        if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
        return number.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function formatMetric(name, value) {
        if (["accuracy", "precision", "recall", "f1_score", "training_accuracy"].includes(name)) {
            return `${(Number(value) * 100).toFixed(2)}%`;
        }
        return formatNumber(value);
    }

    function titleCase(value) {
        return String(value).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
    }

    function clockTime(value = new Date()) {
        const date = value instanceof Date ? value : new Date(value);
        return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString([], { hour12: false });
    }

    function logProgress(message, time = new Date()) {
        const consoleBody = $("#console-body");
        if (!consoleBody) return;
        const line = document.createElement("p");
        line.innerHTML = `<time>${clockTime(time)}</time>${escapeHtml(message)}`;
        consoleBody.appendChild(line);
        consoleBody.scrollTop = consoleBody.scrollHeight;
    }

    async function loadSession() {
        const response = await api("/api/session");
        state.session = response;
        state.dataset = response.active_dataset;
        state.settings = { ...state.settings, ...(response.settings || {}) };
        document.documentElement.dataset.theme = state.settings.theme || "light";
        document.documentElement.dataset.tooltips = state.settings.tooltips === false ? "off" : "on";
        return response;
    }

    function bindGlobalUI() {
        $("#sidebar-toggle")?.addEventListener("click", () => $("#sidebar")?.classList.toggle("open"));
        const shell = $(".app-shell");
        const collapseButton = $("#sidebar-collapse");
        const brand = $(".brand");
        const updateSidebarCollapse = collapsed => {
            shell?.classList.toggle("sidebar-collapsed", collapsed);
            document.documentElement.classList.toggle("sidebar-precollapsed", collapsed);
            collapseButton?.setAttribute("aria-label", collapsed ? "Expand menu" : "Minimize menu");
            if (collapseButton) collapseButton.title = collapsed ? "Expand menu" : "Minimize menu";
        };
        updateSidebarCollapse(localStorage.getItem("sidebarCollapsed") === "true");
        collapseButton?.addEventListener("click", () => {
            const collapsed = !shell.classList.contains("sidebar-collapsed");
            updateSidebarCollapse(collapsed);
            localStorage.setItem("sidebarCollapsed", String(collapsed));
        });
        brand?.addEventListener("click", event => {
            if (!shell?.classList.contains("sidebar-collapsed")) return;
            event.preventDefault();
            updateSidebarCollapse(false);
            localStorage.setItem("sidebarCollapsed", "false");
        });
        $$('[data-sign-out]').forEach(link => link.addEventListener("click", event => {
            event.preventDefault();
            const destination = link.href;
            openModal(`<h2 id="modal-title">Sign Out?</h2><p>Are you sure you want to sign out?</p><div class="button-row modal-footer-actions"><button class="button secondary" data-close-modal>Back</button><button class="button danger" id="confirm-sign-out">Confirm</button></div>`, { hideClose: true });
            $("#confirm-sign-out").addEventListener("click", () => { location.href = destination; });
        }));
        $("#global-modal")?.addEventListener("click", event => {
            if (event.target.id === "global-modal" || event.target.closest("[data-close-modal]")) closeModal();
        });
        document.addEventListener("keydown", event => { if (event.key === "Escape") closeModal(); });
        $("#console-toggle")?.addEventListener("click", () => $("#progress-console")?.classList.toggle("collapsed"));
        $$('[data-password-toggle]').forEach(button => button.addEventListener("click", () => {
            const input = button.closest(".password-field")?.querySelector("input");
            if (!input) return;
            const showing = input.type === "text";
            input.type = showing ? "password" : "text";
            button.setAttribute("aria-label", showing ? "Show password" : "Hide password");
            button.title = showing ? "Show password" : "Hide password";
            button.classList.toggle("showing", !showing);
        }));
    }

    function uploadFile(file, { onProgress } = {}) {
        return new Promise((resolve, reject) => {
            if (!file) return reject(new Error("Choose a file first."));
            if (!/\.(csv|xlsx)$/i.test(file.name)) return reject(new Error("Only CSV and XLSX files are accepted."));
            if (file.size > 1_000_000_000) return reject(new Error("The file exceeds the 1,000 MB limit."));
            const xhr = new XMLHttpRequest();
            const form = new FormData();
            form.append("dataset", file);
            xhr.open("POST", "/api/datasets/upload");
            xhr.upload.onprogress = event => {
                if (event.lengthComputable && onProgress) onProgress(Math.round(event.loaded / event.total * 100));
            };
            xhr.onload = () => {
                let payload;
                try { payload = JSON.parse(xhr.responseText); } catch { payload = { error: `Upload failed (${xhr.status}).` }; }
                if (xhr.status >= 200 && xhr.status < 300 && payload.ok) resolve(payload);
                else reject(new Error(payload.error || "Upload failed."));
            };
            xhr.onerror = () => reject(new Error("The upload connection failed."));
            xhr.send(form);
        });
    }

    // --------------------
    // Comparison Page
    // Handle dataset investigation, model selection, parameters, processing and result charts.
    // --------------------
    async function initComparison() {
        await loadSession();
        const train = $("#train-pct");
        train.value = Math.min(100, Math.max(0, Number(state.settings.train_pct ?? 70)));
        updateSplit();
        train.addEventListener("input", () => setTrainPercentage(train.value));
        $("#train-percent-input").addEventListener("input", event => queueTypedSplit("train", event.target.value));
        $("#test-percent-input").addEventListener("input", event => queueTypedSplit("test", event.target.value));
        [$("#train-percent-input"), $("#test-percent-input")].forEach(input => input.addEventListener("keydown", event => {
            if (event.key === "Enter") { event.preventDefault(); commitTypedSplit(input.id.startsWith("train") ? "train" : "test", input.value); }
        }));

        bindUploadZone();
        bindSlots();
        $("#replace-dataset").addEventListener("click", showDatasetInputOptions);
        $("#import-kaggle-dataset").addEventListener("click", importKaggleDataset);
        $("#kaggle-dataset-url").addEventListener("keydown", event => {
            if (event.key === "Enter") { event.preventDefault(); importKaggleDataset(); }
        });
        $("#column-picker-button").addEventListener("click", showColumnPicker);
        $("#target-select").addEventListener("change", updateDatasetChoices);
        $("#model-search").addEventListener("input", renderModelShelf);
        $("#model-category").addEventListener("change", renderModelShelf);
        $("#export-button").addEventListener("click", showExportModal);
        $("#dock-toggle").addEventListener("click", toggleModelDock);
        $("#parameters-toggle").addEventListener("click", toggleParametersPanel);
        $("#apply-parameters").addEventListener("click", applyParametersAndRun);
        $("#recommend-models").addEventListener("click", recommendStarterModels);
        if (window.ResizeObserver) new ResizeObserver(syncFixedPanelSpacing).observe($("#model-dock"));
        window.addEventListener("resize", syncFixedPanelSpacing);
        requestAnimationFrame(() => {
            syncFixedPanelSpacing();
            requestAnimationFrame(() => $(".comparison-workspace")?.classList.add("comparison-ready"));
        });

        if (state.dataset) renderDataset(state.dataset);
        await loadModels();
        const restored = JSON.parse(localStorage.getItem("restoreComparison") || "null");
        if (restored) {
            localStorage.removeItem("restoreComparison");
            setTrainPercentage(restored.train_pct);
            if (["economy", "balanced", "full"].includes(restored.mode)) state.settings.processing = restored.mode;
            (restored.models || []).slice(0, 2).forEach((saved, index) => {
                const model = state.models.find(item => item.id === saved.id);
                if (model) state.slots[index === 0 ? "a" : "b"] = { ...model, params: { ...(model.defaults || {}), ...(saved.params || {}) } };
            });
            state.activeSlot = state.slots.a ? "a" : state.slots.b ? "b" : null;
        }
        updateSlots();
        const pendingModel = localStorage.getItem("pendingModel");
        if (pendingModel && state.models.some(model => model.id === pendingModel)) {
            localStorage.removeItem("pendingModel");
            chooseModel(pendingModel);
        }
    }

    function updateSplit() {
        const value = Number($("#train-pct").value);
        $("#train-percent-input").value = value;
        $("#test-percent-input").value = 100 - value;
        $("#train-bar").style.width = `${value}%`;
        $("#test-bar").style.width = `${100 - value}%`;
        $("#train-bar").textContent = value >= 24 ? `Train ${value}%` : "";
        $("#test-bar").textContent = 100 - value >= 24 ? `Test ${100 - value}%` : "";
        $("#split-divider").style.left = `${value}%`;
        const warning = $("#split-warning");
        const endpoint = value === 0 || value === 100;
        const extreme = value < 10 || value > 90;
        warning.classList.toggle("hidden", !extreme);
        warning.textContent = endpoint
            ? "This split is allowed for checking the percentage control, but modelling needs at least 1% training data and 1% testing data."
            : extreme
                ? "A very small training or testing portion can produce unreliable results, and some models may fail when too few rows or classes are available."
                : "";
    }

    function setTrainPercentage(rawValue, runComparison = true) {
        if (rawValue === "" || Number.isNaN(Number(rawValue))) return;
        const value = Math.min(100, Math.max(0, Math.round(Number(rawValue))));
        $("#train-pct").value = value;
        updateSplit();
        if (runComparison) maybeRunComparison();
    }

    function queueTypedSplit(source, rawValue) {
        clearTimeout(splitInputTimer);
        const text = String(rawValue).trim();
        if (/^\d+$/.test(text) && Number(text) >= 0 && Number(text) <= 100) {
            const entered = Number(text);
            setTrainPercentage(source === "test" ? 100 - entered : entered, false);
            splitInputTimer = setTimeout(maybeRunComparison, 650);
            return;
        }
        splitInputTimer = setTimeout(() => commitTypedSplit(source, rawValue), 800);
    }

    function commitTypedSplit(source, rawValue) {
        clearTimeout(splitInputTimer);
        const text = String(rawValue).trim();
        if (!/^\d+$/.test(text) || Number(text) < 0 || Number(text) > 100) {
            updateSplit();
            toast("Enter a whole number from 0 to 100.", "warning");
            return;
        }
        const entered = Number(text);
        setTrainPercentage(source === "test" ? 100 - entered : entered);
    }

    function toggleModelDock() {
        const dock = $("#model-dock");
        const collapsed = dock.classList.toggle("collapsed");
        const button = $("#dock-toggle");
        button.setAttribute("aria-expanded", String(!collapsed));
        button.setAttribute("aria-label", collapsed ? "Expand model shelf" : "Minimize model shelf");
        button.title = collapsed ? "Expand model shelf" : "Minimize model shelf";
        requestAnimationFrame(syncFixedPanelSpacing);
    }

    function syncFixedPanelSpacing() {
        const dock = $("#model-dock");
        const workspace = $(".comparison-workspace");
        if (dock && workspace) workspace.style.setProperty("--dock-clearance", `${Math.ceil(dock.getBoundingClientRect().height) + 22}px`);
    }

    function toggleParametersPanel() {
        const panel = $("#parameters-panel");
        const collapsed = panel.classList.toggle("collapsed");
        $(".comparison-workspace")?.classList.toggle("parameters-collapsed", collapsed);
        const button = $("#parameters-toggle");
        button.setAttribute("aria-expanded", String(!collapsed));
        button.setAttribute("aria-label", collapsed ? "Expand parameters" : "Minimize parameters");
        button.title = collapsed ? "Expand parameters" : "Minimize parameters";
    }

    function bindUploadZone() {
        const input = $("#dataset-upload");
        const zone = $("#comparison-upload-zone");
        input.addEventListener("change", () => handleComparisonUpload(input.files[0]));
        ["dragenter", "dragover"].forEach(type => zone.addEventListener(type, event => { event.preventDefault(); zone.classList.add("dragging"); }));
        ["dragleave", "drop"].forEach(type => zone.addEventListener(type, event => { event.preventDefault(); zone.classList.remove("dragging"); }));
        zone.addEventListener("drop", event => handleComparisonUpload(event.dataTransfer.files[0]));
    }

    function showDatasetInputOptions() {
        $("#dataset-card").classList.add("hidden");
        $("#dataset-input-options").classList.remove("hidden");
        $("#dataset-upload").value = "";
        $("#kaggle-dataset-url").focus();
    }

    async function useInvestigatedDataset(response, progressMessage, successMessage) {
        state.dataset = response.dataset;
        logProgress(progressMessage);
        renderDataset(state.dataset);
        await loadModels();
        updateSlots();
        toast(successMessage);
        maybeRunComparison();
    }

    async function handleComparisonUpload(file) {
        if (!file) return;
        const progress = $("#upload-progress");
        progress.classList.remove("hidden");
        logProgress(`Uploading ${file.name} (${(file.size / 1_000_000).toFixed(2)} MB)`);
        try {
            const response = await uploadFile(file, { onProgress: percent => {
                $("#dataset-status").textContent = `Uploading ${percent}%`;
            }});
            await useInvestigatedDataset(
                response,
                `Upload complete. Profiled ${response.dataset.profiled_rows.toLocaleString()} row(s).`,
                "Dataset uploaded and investigated.",
            );
        } catch (error) {
            logProgress(`Upload stopped: ${error.message}`);
            toast(error.message, "error");
        } finally {
            progress.classList.add("hidden");
        }
    }

    async function importKaggleDataset() {
        const input = $("#kaggle-dataset-url");
        const button = $("#import-kaggle-dataset");
        const status = $("#kaggle-dataset-status");
        const datasetUrl = input.value.trim();
        if (!datasetUrl) { toast("Paste a Kaggle dataset URL first.", "warning"); input.focus(); return; }
        button.disabled = true;
        button.textContent = "Importing…";
        status.className = "kaggle-dataset-status";
        status.textContent = "Downloading the public dataset from Kaggle and checking its CSV or XLSX files…";
        logProgress("Importing a public Kaggle dataset. Larger datasets may take longer to download.");
        try {
            const response = await api("/api/datasets/kaggle", { method: "POST", body: { url: datasetUrl } });
            await useInvestigatedDataset(
                response,
                `Kaggle import complete. Selected ${response.dataset.filename} and profiled ${response.dataset.profiled_rows.toLocaleString()} row(s).`,
                "Kaggle dataset imported and investigated.",
            );
        } catch (error) {
            status.className = "kaggle-dataset-status error-text";
            status.textContent = error.message;
            logProgress(`Kaggle import stopped: ${error.message}`);
            toast(error.message, "error", 7000);
        } finally {
            button.disabled = false;
            button.textContent = "Import from Kaggle";
        }
    }

    function renderDataset(dataset) {
        $("#dataset-input-options").classList.add("hidden");
        $("#dataset-card").classList.remove("hidden");
        $("#dataset-name").textContent = dataset.filename;
        $("#dataset-status").textContent = `${dataset.task === "unknown" ? "Target needs attention" : titleCase(dataset.task)} · ${dataset.profile_seconds}s profiling`;
        const stats = [
            ["File size", `${formatNumber(dataset.size_mb, 2)} MB`],
            ["Rows", Number(dataset.rows).toLocaleString()],
            ["Columns", Number(dataset.columns).toLocaleString()],
            ["Missing cells", `${Number(dataset.missing_cells).toLocaleString()} (${dataset.missing_pct}%)`],
            ["Duplicate rows", Number(dataset.duplicate_rows).toLocaleString()],
        ];
        $("#dataset-stats").innerHTML = stats.map(([label, value]) => `<div class="stat-item"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`).join("");
        $("#target-select").innerHTML = dataset.column_names.map(name => `<option value="${escapeHtml(name)}" ${name === dataset.target ? "selected" : ""}>${escapeHtml(name)}</option>`).join("");
        $("#target-explanation").innerHTML = `<strong>${dataset.target_confidence === "manual" ? "Manual target" : `${titleCase(dataset.target_confidence)}-confidence suggestion`}: ${escapeHtml(dataset.target)}</strong><br>${escapeHtml(dataset.target_reason)}. ${escapeHtml(dataset.task_reason)}`;
        if ($("#recommend-models")) $("#recommend-models").disabled = dataset.task === "unknown";
        const visibleColumns = (dataset.selected_columns || dataset.column_names).filter(name => dataset.column_names.includes(name));
        $("#column-summary-table tbody").innerHTML = dataset.column_summaries.filter(column => visibleColumns.includes(column.name)).map(column => {
            const main = column.mean !== undefined ? formatNumber(column.mean) : (column.top_values?.[0] ? `${escapeHtml(column.top_values[0].value)} (${column.top_values[0].count})` : "—");
            return `<tr><td><strong>${escapeHtml(column.name)}</strong></td><td>${escapeHtml(column.dtype)}</td><td>${column.missing.toLocaleString()} (${column.missing_pct}%)</td><td>${column.unique.toLocaleString()}</td><td>${main}</td><td>${column.std === undefined ? "—" : formatNumber(column.std)}</td></tr>`;
        }).join("");
        renderTable($("#dataset-preview-table"), dataset.preview, visibleColumns);
    }

    function renderTable(table, records, columns = null) {
        if (!table) return;
        const names = columns || (records?.length ? Object.keys(records[0]) : []);
        if (!records?.length) { table.innerHTML = `<tbody><tr><td>No preview rows are available.</td></tr></tbody>`; return; }
        table.innerHTML = `<thead><tr>${names.map(name => `<th>${escapeHtml(name)}</th>`).join("")}</tr></thead><tbody>${records.map(record => `<tr>${names.map(name => `<td>${escapeHtml(record[name] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }

    async function updateDatasetChoices() {
        if (!state.dataset) return;
        try {
            const response = await api(`/api/datasets/${state.dataset.id}`, {
                method: "PATCH",
                body: { target: $("#target-select").value },
            });
            state.dataset = response.dataset;
            renderDataset(state.dataset);
            await loadModels();
            updateSlots();
            logProgress(`Target updated to '${state.dataset.target}'. Detected ${state.dataset.task}.`);
            maybeRunComparison();
        } catch (error) { toast(error.message, "error"); }
    }

    function showColumnPicker() {
        if (!state.dataset) return;
        openModal(`<h2 id="modal-title">Choose Model Input Columns</h2><p>Untick identifiers, notes, or other noise. The target remains included automatically.</p><div class="column-checks">${state.dataset.column_names.map(name => `<label><input type="checkbox" value="${escapeHtml(name)}" ${(state.dataset.selected_columns || []).includes(name) ? "checked" : ""} ${name === state.dataset.target ? "disabled" : ""}>${escapeHtml(name)}${name === state.dataset.target ? " (target)" : ""}</label>`).join("")}</div><button class="button primary" id="save-column-choice">Use Selected Columns</button>`);
        $("#save-column-choice").addEventListener("click", async () => {
            const selected = $$(".column-checks input:checked").map(input => input.value);
            try {
                const response = await api(`/api/datasets/${state.dataset.id}`, { method: "PATCH", body: { selected_columns: selected } });
                state.dataset = response.dataset;
                closeModal();
                renderDataset(state.dataset);
                await loadModels();
                updateSlots();
                toast(`${selected.length} columns selected for modelling.`);
                logProgress(`Input selection updated: ${selected.length - 1} feature column(s) plus the target.`);
                maybeRunComparison();
            } catch (error) { toast(error.message, "error"); }
        });
    }

    async function loadModels() {
        const response = await api("/api/models");
        state.models = response.models;
        // Refresh slot objects so compatibility follows a newly selected target.
        for (const key of ["a", "b"]) {
            if (state.slots[key]) {
                const fresh = state.models.find(model => model.id === state.slots[key].id);
                if (fresh) {
                    const params = { ...(fresh.defaults || {}), ...(state.slots[key].params || {}) };
                    (fresh.parameter_schema || []).forEach(schema => {
                        if (Array.isArray(schema.choices) && !schema.choices.map(String).includes(String(params[schema.name]))) params[schema.name] = fresh.defaults?.[schema.name];
                    });
                    state.slots[key] = { ...fresh, params };
                } else state.slots[key] = null;
            }
        }
        if (!state.activeSlot || !state.slots[state.activeSlot]) {
            state.activeSlot = state.slots.a ? "a" : state.slots.b ? "b" : null;
        }
        renderModelShelf();
    }

    async function recommendStarterModels() {
        if (!state.dataset) return toast("Upload or select a dataset first.", "warning");
        const button = $("#recommend-models");
        button.disabled = true;
        button.textContent = "Finding Compatible Models…";
        try {
            const recommendation = await api("/api/models/recommendations");
            if (!recommendation.models.length) throw new Error("No starter recommendation is available for this target. Choose a model manually.");
            state.slots.a = null; state.slots.b = null;
            recommendation.models.forEach((suggestion, index) => {
                const model = state.models.find(item => item.id === suggestion.id);
                if (model) state.slots[index === 0 ? "a" : "b"] = { ...model, params: { ...(model.defaults || {}), ...(suggestion.defaults || {}) } };
            });
            state.activeSlot = state.slots.a ? "a" : null;
            const explanation = $("#recommendation-explanation");
            explanation.classList.remove("hidden");
            explanation.innerHTML = `<strong>${escapeHtml(recommendation.message)}</strong><div class="recommendation-reasons">${recommendation.models.map(model => `<span><b>${escapeHtml(model.name)}</b>${escapeHtml(model.reason)}</span>`).join("")}</div><p>${escapeHtml(recommendation.disclaimer)}</p>`;
            updateSlots(); renderModelShelf();
            logProgress(`Recommended ${recommendation.models.map(model => model.name).join(" and ")} for this ${recommendation.task} task. Manual controls remain available.`);
            maybeRunComparison(true);
        } catch (error) { toast(error.message, "error"); }
        finally { button.disabled = state.dataset?.task === "unknown"; button.textContent = "Recommend Models for Me"; }
    }

    function renderModelShelf() {
        const list = $("#model-list");
        if (!list) return;
        const query = ($("#model-search")?.value || "").trim().toLowerCase();
        const category = $("#model-category")?.value || "all";
        const recentIds = JSON.parse(localStorage.getItem("recentModels") || "[]");
        const filtered = state.models.filter(model => {
            const match = !query || [model.name, model.family, model.summary, model.best_for].join(" ").toLowerCase().includes(query);
            const group = category === "all" || (category === "custom" && model.id.startsWith("custom:")) || (category === "library" && !model.id.startsWith("custom:")) || (category === "recent" && recentIds.includes(model.id));
            return match && group;
        });
        if (category === "recent") filtered.sort((a, b) => recentIds.indexOf(a.id) - recentIds.indexOf(b.id));
        list.innerHTML = filtered.length ? filtered.map(model => { const guidance = `${model.summary} Best for: ${model.best_for} ${model.compatibility_reason || ""}`; return `<article class="model-card ${model.compatible === false ? "incompatible" : ""}" draggable="${model.compatible !== false}" data-model-id="${escapeHtml(model.id)}" tabindex="0" title="${escapeHtml(guidance)}" data-tooltip="${escapeHtml(guidance)}"><span class="compatibility ${model.compatible === true ? "yes" : model.compatible === false ? "no" : ""}"></span><span class="family">${escapeHtml(model.family)}</span><h3>${escapeHtml(model.name)}</h3><p>${escapeHtml(model.summary)}</p><div class="best-for"><strong>Best For:</strong> ${escapeHtml(model.best_for)}</div></article>`; }).join("") : `<div class="empty-card">No models match this filter.</div>`;
        $$(".model-card", list).forEach(card => {
            card.addEventListener("dragstart", event => event.dataTransfer.setData("text/model-id", card.dataset.modelId));
            card.addEventListener("click", () => chooseModel(card.dataset.modelId));
            card.addEventListener("keydown", event => { if (event.key === "Enter") chooseModel(card.dataset.modelId); });
        });
    }

    function bindSlots() {
        $$(".model-slot").forEach(slot => {
            slot.addEventListener("click", event => {
                if (event.target.closest("[data-remove-slot]")) return;
                if (state.slots[slot.dataset.slot]) {
                    state.activeSlot = slot.dataset.slot;
                    updateSlots(state.result?.models || null);
                }
            });
            slot.addEventListener("dragover", event => { event.preventDefault(); slot.classList.add("drag-over"); });
            slot.addEventListener("dragleave", () => slot.classList.remove("drag-over"));
            slot.addEventListener("drop", event => {
                event.preventDefault(); slot.classList.remove("drag-over");
                assignModel(slot.dataset.slot, event.dataTransfer.getData("text/model-id"));
            });
        });
    }

    function chooseModel(modelId) {
        const model = state.models.find(item => item.id === modelId);
        if (!model) return;
        if (model.compatible === false && state.dataset) {
            toast(model.compatibility_reason, "warning");
            return;
        }
        if (!state.slots.a) assignModel("a", modelId);
        else if (!state.slots.b) assignModel("b", modelId);
        else {
            openModal(`<h2 id="modal-title">Replace a Comparison Model</h2><p>Both slots are occupied. Choose which card should be replaced with <strong>${escapeHtml(model.name)}</strong>.</p><div class="button-row"><button class="button secondary" data-replace-slot="a">Replace Model A</button><button class="button secondary" data-replace-slot="b">Replace Model B</button></div>`);
            $$('[data-replace-slot]').forEach(button => button.addEventListener("click", () => { closeModal(); assignModel(button.dataset.replaceSlot, modelId); }));
        }
    }

    function assignModel(slotName, modelId) {
        const model = state.models.find(item => item.id === modelId);
        if (!model) return;
        if (state.dataset && model.compatible === false) { toast(model.compatibility_reason, "warning"); return; }
        state.slots[slotName] = { ...model, params: { ...(model.defaults || {}) } };
        $("#recommendation-explanation")?.classList.add("hidden");
        state.activeSlot = slotName;
        const recent = [model.id, ...JSON.parse(localStorage.getItem("recentModels") || "[]").filter(id => id !== model.id)].slice(0, 8);
        localStorage.setItem("recentModels", JSON.stringify(recent));
        updateSlots(); renderModelShelf();
        logProgress(`${model.name} placed in Model ${slotName.toUpperCase()}.`);
        maybeRunComparison();
    }

    function updateSlots(results = null) {
        for (const key of ["a", "b"]) {
            const slot = $(`#slot-${key}`);
            const model = state.slots[key];
            const result = results?.find(item => item.model_id === model?.id);
            if (!model) {
                slot.classList.remove("selected");
                slot.innerHTML = `<div class="slot-empty"><span>${key.toUpperCase()}</span><strong>Drop Model ${key.toUpperCase()} Here</strong><small>${key === "b" ? "One model is enough to begin." : "You can also click a model below."}</small></div>`;
                continue;
            }
            slot.classList.toggle("selected", state.activeSlot === key);
            const warning = state.dataset && model.compatible === false ? model.compatibility_reason : null;
            const metricRows = result ? Object.entries(result.metrics).map(([name, value]) => `<div class="metric-row"><span>${escapeHtml(titleCase(name))}</span><strong>${escapeHtml(formatMetric(name, value))}</strong></div>`).join("") + `<div class="metric-row"><span>Training Time</span><strong>${formatNumber(result.training_seconds)} s</strong></div>` : `<div class="metric-row"><span>Parameters</span><strong>${Object.keys(model.params || {}).length} defaults</strong></div><div class="metric-row"><span>Status</span><strong>${warning ? "Waiting" : "Ready"}</strong></div>`;
            slot.innerHTML = `<div class="model-result-head"><div><span class="family">${escapeHtml(model.family)}</span><h3>${escapeHtml(model.name)} <button class="info-button" data-tooltip="${escapeHtml(`${model.summary} Best for: ${model.best_for}`)}">i</button></h3><p>${escapeHtml(result ? `${titleCase(result.task)} Result` : model.compatibility_reason || "Waiting for data")}</p></div><div><span class="slot-badge">Model ${key.toUpperCase()}</span><button class="remove-model" data-remove-slot="${key}" aria-label="Remove model">×</button></div></div><div class="model-metrics">${metricRows}</div>${warning ? `<div class="slot-warning">${escapeHtml(warning)}</div>` : ""}`;
        }
        $$('[data-remove-slot]').forEach(button => button.addEventListener("click", () => {
            const removed = button.dataset.removeSlot;
            state.slots[removed] = null; state.result = null;
            if (state.activeSlot === removed) state.activeSlot = state.slots.a ? "a" : state.slots.b ? "b" : null;
            $("#results-area").classList.add("hidden"); $("#export-button").disabled = true;
            updateSlots(); updateComparisonNotice();
        }));
        renderParameterPanel();
        updateComparisonNotice();
    }

    function renderParameterPanel() {
        const model = state.activeSlot ? state.slots[state.activeSlot] : null;
        const root = $("#parameter-editor-panel");
        if (!root) return;
        if (!model) {
            $("#parameters-model-name").textContent = "NO MODEL SELECTED";
            $("#parameters-help").textContent = "Choose a model card to view its default parameters.";
            root.innerHTML = `<div class="empty-parameter-state">No model selected.</div>`;
            $("#parameters-footer").classList.add("hidden");
            return;
        }
        $("#parameters-model-name").textContent = model.name.toUpperCase();
        $("#parameters-model-name").title = model.name;
        $("#parameters-footer").classList.remove("hidden");
        $("#parameters-help").textContent = `Model ${state.activeSlot.toUpperCase()} · ${model.name}. Defaults are ready to run.`;
        const entries = Object.entries(model.params || {});
        const schemaByName = Object.fromEntries((model.parameter_schema || []).map(item => [item.name, item]));
        root.innerHTML = entries.length ? entries.map(([name, value]) => {
            const schema = schemaByName[name] || {};
            const type = schema.type || typeof value;
            const label = schema.label || titleCase(name);
            const attributes = (schema.min !== undefined ? ` min="${escapeHtml(schema.min)}"` : "") + (schema.max !== undefined ? ` max="${escapeHtml(schema.max)}"` : "");
            if (Array.isArray(schema.choices)) return `<label>${escapeHtml(label)}<select data-panel-param-name="${escapeHtml(name)}" data-param-type="text">${schema.choices.map(choice => `<option value="${escapeHtml(choice)}" ${String(value) === String(choice) ? "selected" : ""}>${escapeHtml(titleCase(choice))}</option>`).join("")}</select><small>${escapeHtml(schema.description || "Uses the model library default.")}</small></label>`;
            if (type === "boolean") return `<label>${escapeHtml(label)}<select data-panel-param-name="${escapeHtml(name)}" data-param-type="boolean"><option value="true" ${value === true ? "selected" : ""}>True</option><option value="false" ${value === false ? "selected" : ""}>False</option></select><small>${escapeHtml(schema.description || "Uses the model library default.")}</small></label>`;
            return `<label>${escapeHtml(label)}<input data-panel-param-name="${escapeHtml(name)}" value="${escapeHtml(Array.isArray(value) ? value.join(",") : value)}" data-param-type="${escapeHtml(type)}" type="${["integer", "number"].includes(type) ? "number" : "text"}"${attributes}><small>${escapeHtml(schema.description || "Uses the model library default.")}</small></label>`;
        }).join("") : `<div class="empty-parameter-state">This model has no editable parameters.</div>`;
    }

    function applyParametersAndRun() {
            const model = state.activeSlot ? state.slots[state.activeSlot] : null;
            const root = $("#parameter-editor-panel");
            if (!model || !root) return;
            $$('[data-panel-param-name]', root).forEach(input => {
                const name = input.dataset.panelParamName;
                const original = model.params[name];
                let value = input.value;
                if (Array.isArray(original)) value = input.value.split(",").map(Number);
                else if (["integer", "number"].includes(input.dataset.paramType) || typeof original === "number") value = Number(input.value);
                else if (input.dataset.paramType === "boolean" || typeof original === "boolean") value = ["true", "1", "yes"].includes(input.value.toLowerCase());
                model.params[name] = value;
            });
            toast(`${model.name} parameters updated.`); updateSlots(); maybeRunComparison();
    }

    function selectedModels() { return [state.slots.a, state.slots.b].filter(Boolean); }

    function updateComparisonNotice(message = null) {
        const notice = $("#comparison-notice");
        const models = selectedModels();
        notice.classList.remove("important");
        if (message) { notice.textContent = message; notice.classList.remove("hidden"); return; }
        if (!state.dataset && models.length) { notice.textContent = "A dataset is required before model statistics can be calculated."; notice.classList.add("important"); }
        else if (state.dataset && !models.length) notice.textContent = "Your dataset is ready. Add one or two compatible models from the shelf.";
        else if (!state.dataset) notice.textContent = "Upload a dataset and add a model to generate real results.";
        else notice.textContent = "Ready. The comparison will start automatically.";
    }

    let runDebounce;
    function maybeRunComparison(showGuidedBudgetChoice = false) {
        clearTimeout(runDebounce);
        if (!state.dataset || !selectedModels().length) { updateComparisonNotice(); return; }
        const trainPercentage = Number($("#train-pct").value);
        if (trainPercentage === 0 || trainPercentage === 100) {
            updateComparisonNotice("Choose at least 1% training data and 1% testing data before running models.");
            return;
        }
        if (selectedModels().some(model => model.compatible === false)) { updateComparisonNotice("Replace incompatible models before running."); return; }
        const tasks = selectedModels().map(model => model.tasks.includes("clustering") ? "clustering" : state.dataset.task);
        if (new Set(tasks).size > 1) { updateComparisonNotice("These models solve different task types. Compare models from the same category."); return; }
        runDebounce = setTimeout(() => runPreflight(showGuidedBudgetChoice), 450);
    }

    async function runPreflight(showGuidedBudgetChoice = false) {
        const models = selectedModels().map(model => ({ id: model.id, name: model.name, params: model.params }));
        try {
            const preflight = await api("/api/comparisons/preflight", { method: "POST", body: { models } });
            logProgress(`Preflight estimate: approximately ${preflight.estimated_seconds.toFixed(1)} second(s).`);
            if (showGuidedBudgetChoice) showProcessingChoice(preflight);
            else {
                const selected = preflight.modes.find(mode => mode.id === state.processingMode) || preflight.modes.find(mode => mode.id === "economy");
                logProgress(`Running directly with the current ${selected.label} budget: up to ${Number(selected.rows).toLocaleString()} rows.`);
                startComparison(selected.id);
            }
        } catch (error) { toast(error.message, "error"); logProgress(error.message); }
    }

    function showProcessingChoice(preflight) {
        const economyRows = Number(preflight.modes.find(mode => mode.id === "economy")?.rows || 0);
        const purpose = {
            economy: ["Quick exploration", "Low resource use", "Up to 10,000 rows"],
            balanced: ["Broader comparison", "Medium resource use", "Up to 50,000 rows"],
            full: ["Final detailed run", "Highest resource use", "Every available row"],
        };
        const options = preflight.modes.map(mode => {
            const details = purpose[mode.id];
            const sameRows = mode.id !== "economy" && Number(mode.rows) === economyRows;
            return `<label class="modal-option budget-option budget-${escapeHtml(mode.id)}"><input type="radio" name="processing-mode" value="${escapeHtml(mode.id)}" ${state.processingMode === mode.id ? "checked" : ""}><span><strong>${escapeHtml(mode.label)}${mode.id === "economy" ? " · Default" : ""}</strong><span class="budget-facts"><b>${escapeHtml(details[0])}</b><b>${escapeHtml(details[1])}</b><b>${escapeHtml(details[2])}</b></span><small><strong>${Number(mode.rows).toLocaleString()} of ${Number(preflight.full_rows).toLocaleString()} rows</strong> will be processed · estimated ${Number(mode.estimated_seconds).toFixed(1)} seconds.</small>${sameRows ? `<em class="same-budget-warning">Same row count as Economy for this dataset, so results may be identical.</em>` : `<em>${escapeHtml(mode.tradeoff)}</em>`}</span></label>`;
        }).join("");
        openModal(`<h2 id="modal-title">Choose a Resource Budget</h2><p><strong>Budgets change the number of rows processed—not the selected models.</strong> Larger datasets make the differences more visible.</p><div class="modal-options">${options}</div><div class="modal-guidance"><strong>You remain in control.</strong> Your latest choice is remembered for comparisons on this page. Refreshing or closing the page resets it to Economy.</div><button class="button primary" id="begin-processing">Begin Processing</button>`);
        $("#begin-processing").addEventListener("click", () => {
            const mode = $('input[name="processing-mode"]:checked').value;
            state.processingMode = mode;
            closeModal(); startComparison(mode);
        });
    }

    async function startComparison(mode) {
        const models = selectedModels().map(model => ({ id: model.id, name: model.name, params: model.params }));
        updateComparisonNotice("Processing is underway. Follow each stage in the progress console.");
        $("#console-status").textContent = "Running";
        state.lastLogs = 0;
        logProgress(`Starting ${mode} processing with ${models.map(model => model.name).join(" and ")}.`);
        try {
            const response = await api("/api/comparisons", {
                method: "POST",
                body: { models, mode, train_pct: Number($("#train-pct").value), seed: Number(state.settings.random_seed || 42) },
            });
            pollJob(response.job_id);
        } catch (error) { toast(error.message, "error"); updateComparisonNotice(error.message); }
    }

    async function pollJob(jobId) {
        try {
            const response = await api(`/api/jobs/${jobId}`);
            const job = response.job;
            (job.logs || []).slice(state.lastLogs).forEach(entry => logProgress(entry.message, entry.time));
            state.lastLogs = (job.logs || []).length;
            $("#console-status").textContent = `${job.progress || 0}% · ${titleCase(job.status)}`;
            if (job.status === "running") setTimeout(() => pollJob(jobId), 650);
            else if (job.status === "failed") {
                toast(job.error, "error"); updateComparisonNotice(job.error); $("#console-status").textContent = "Failed";
            } else {
                state.result = job.result;
                renderResults(job.result);
                $("#console-status").textContent = "Complete";
                toast("Model comparison completed.");
            }
        } catch (error) { toast(error.message, "error"); }
    }

    function renderResults(result) {
        updateSlots(result.models);
        $("#comparison-notice").classList.add("hidden");
        $("#results-area").classList.remove("hidden");
        $("#export-button").disabled = false;
        const receipt = result.affordability || {};
        $("#result-summary").textContent = `${titleCase(result.task)} · ${result.train_pct}% train / ${result.test_pct}% test · ${receipt.resource_mode_label || titleCase(result.mode || "full")} mode · ${Number(result.rows_processed).toLocaleString()} rows processed`;
        $("#result-id").textContent = `Result ${result.id.slice(0, 8)}`;
        $("#transformation-note").innerHTML = `<strong>Automatic preparation used for modelling</strong><br>${(result.transformations || []).map(escapeHtml).join(" · ")}`;
        const accessItems = [
            ["Application licence/API fee", `RM${Number(receipt.application_fee_myr || 0).toFixed(2)}`],
            ["Paid external ML calls", Number(receipt.paid_external_api_calls || 0).toLocaleString()],
            ["Processing location", receipt.processing_location || "Local computer"],
            ["Enterprise software", receipt.enterprise_software_required ? "Required" : "Not required"],
            ["Resource budget", receipt.resource_mode_label || titleCase(result.mode || "full")],
            ["Rows processed", `${Number(result.rows_processed).toLocaleString()} / ${Number(result.rows_available || result.rows_processed).toLocaleString()}`],
            ["Measured training time", `${formatNumber(receipt.training_seconds ?? result.total_training_seconds ?? 0)} s`],
        ];
        $("#affordability-receipt").innerHTML = accessItems.map(([label, value]) => `<div class="receipt-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
        $("#receipt-note").textContent = receipt.note || "No application fee was charged for this local run. Ordinary device, electricity, and internet costs are not estimated.";
        renderMetricBars(result);
        renderDetailCharts(result);
    }

    function renderMetricBars(result) {
        const chart = $("#metric-chart");
        const metrics = result.comparison_chart.labels;
        chart.innerHTML = metrics.map((metric, metricIndex) => {
            const values = result.comparison_chart.series.map(series => Number(series.values[metricIndex])).filter(Number.isFinite);
            const max = Math.max(...values.map(Math.abs), 1);
            const lines = result.comparison_chart.series.map(series => {
                const value = series.values[metricIndex];
                if (value === null || value === undefined) return "";
                const width = Math.max(1, Math.min(100, Math.abs(Number(value)) / max * 100));
                return `<div class="bar-line"><span>${escapeHtml(series.name)}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${escapeHtml(formatMetric(metric, value))}</strong></div>`;
            }).join("");
            return `<div class="chart-row"><span class="chart-row-label">${escapeHtml(titleCase(metric))}</span><div class="bar-group">${lines}</div></div>`;
        }).join("");
    }

    function renderDetailCharts(result) {
        const container = $("#detail-charts");
        container.innerHTML = result.models.map((model, index) => {
            if (model.chart?.type === "confusion") {
                const labels = model.chart.labels || [];
                const maximum = Math.max(...(model.chart.matrix || []).flat(), 1);
                const header = labels.map(label => `<th scope="col">${escapeHtml(label)}</th>`).join("");
                const rows = (model.chart.matrix || []).map((row, rowIndex) => `<tr><th scope="row">${escapeHtml(labels[rowIndex])}</th>${row.map(value => `<td style="--cell-strength:${Math.max(8, Number(value) / maximum * 100)}%">${escapeHtml(value)}</td>`).join("")}</tr>`).join("");
                return `<article class="detail-card"><h4>${escapeHtml(model.name)} · Confusion Matrix</h4><div class="confusion-axis predicted-axis">Predicted Class →</div><div class="confusion-layout"><div class="confusion-axis actual-axis">Actual Class ↓</div><div class="confusion-table-wrap"><table class="confusion-table"><thead><tr><th scope="col">Actual \\ Predicted</th>${header}</tr></thead><tbody>${rows}</tbody></table></div></div></article>`;
            }
            return `<article class="detail-card"><h4>${escapeHtml(model.name)} · ${escapeHtml(titleCase(model.chart?.type || "distribution"))}</h4><canvas class="mini-chart" data-chart-index="${index}" width="580" height="220"></canvas></article>`;
        }).join("");
        $$("canvas[data-chart-index]", container).forEach(canvas => drawModelChart(canvas, result.models[Number(canvas.dataset.chartIndex)], Number(canvas.dataset.chartIndex)));
    }

    function drawModelChart(canvas, model, index) {
        const ctx = canvas.getContext("2d");
        const width = canvas.width, height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--border-strong");
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted");
        ctx.font = "12px system-ui";
        const chart = model.chart || {};
        if (chart.type === "confusion") {
            const matrix = chart.matrix, n = matrix.length, max = Math.max(...matrix.flat(), 1), size = Math.min(38, 150 / Math.max(n, 1));
            matrix.forEach((row, y) => row.forEach((value, x) => {
                const alpha = .15 + .8 * value / max;
                ctx.fillStyle = index ? `rgba(124,58,237,${alpha})` : `rgba(49,87,213,${alpha})`;
                ctx.fillRect(60 + x * size, 25 + y * size, size - 2, size - 2);
                ctx.fillStyle = "#ffffff"; ctx.fillText(String(value), 66 + x * size, 43 + y * size);
            }));
            ctx.fillStyle = "#667085"; ctx.fillText("Rows: actual · Columns: predicted", 60, 205);
            return;
        }
        if (chart.values) {
            const max = Math.max(...chart.values, 1), gap = 12, barWidth = Math.max(10, (width - 90) / chart.values.length - gap);
            chart.values.forEach((value, i) => {
                const barHeight = value / max * 145;
                ctx.fillStyle = index ? "#7c3aed" : "#3157d5";
                ctx.fillRect(50 + i * (barWidth + gap), 180 - barHeight, barWidth, barHeight);
                ctx.fillStyle = "#667085"; ctx.fillText(chart.labels[i], 50 + i * (barWidth + gap), 202);
            });
            return;
        }
        const actual = chart.actual || [], predicted = chart.predicted || [];
        const all = [...actual, ...predicted].map(Number).filter(Number.isFinite);
        if (!all.length) { ctx.fillText("No chart points available.", 30, 50); return; }
        const min = Math.min(...all), max = Math.max(...all), span = max - min || 1;
        const drawLine = (values, color) => {
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
            values.forEach((value, i) => {
                const x = 35 + i / Math.max(1, values.length - 1) * (width - 60);
                const y = 185 - (Number(value) - min) / span * 150;
                i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }); ctx.stroke();
        };
        drawLine(actual, "#3157d5"); drawLine(predicted, "#7c3aed");
        ctx.fillStyle = "#3157d5"; ctx.fillText("Actual", 35, 210); ctx.fillStyle = "#7c3aed"; ctx.fillText("Predicted", 95, 210);
    }

    function showExportModal() {
        if (!state.result) return;
        openModal(`<h2 id="modal-title">Export Comparison Result</h2><p>Choose a format for result ${escapeHtml(state.result.id.slice(0, 8))}.</p><div class="modal-options"><a class="modal-option" href="/api/results/${state.result.id}/export.pdf"><span><strong>PDF Report</strong><small>Summary and model metrics for academic reporting.</small></span></a><a class="modal-option" href="/api/results/${state.result.id}/export.png"><span><strong>PNG Image</strong><small>A portable visual summary.</small></span></a><a class="modal-option" href="/api/results/${state.result.id}/export.xlsx"><span><strong>XLSX Workbook</strong><small>Summary, metrics, and selected-column sheets.</small></span></a></div>`);
    }

    // --------------------
    // Data Cleaning Page
    // Handle cleaning actions, joined datasets, downloads and sending a clean copy to comparison.
    // --------------------
    const cleaningState = { datasets: [], active: null, second: null, operation: "remove_missing", output: null, outputOperation: null, outputSourceId: null, removedDuplicates: [], customActions: [], editingCustomId: null };

    async function initCleaning() {
        await loadSession();
        await refreshCleaningDatasets();
        $$(".tool-button").forEach(button => button.addEventListener("click", () => chooseCleaningOperation(button.dataset.operation)));
        $("#cleaning-dataset-select").addEventListener("change", async event => {
            await api(`/api/datasets/${event.target.value}/activate`, { method: "POST" });
            resetCleaningOutput();
            cleaningState.active = cleaningState.datasets.find(item => item.id === event.target.value);
            renderCleaningSource(); renderCleaningFields();
        });
        $("#cleaning-form").addEventListener("submit", runCleaning);
        $("#cleaning-upload-button").addEventListener("click", () => $("#cleaning-upload").click());
        $("#delete-cleaning-dataset").addEventListener("click", confirmDeleteCleaningDataset);
        $("#delete-all-cleaning-datasets")?.addEventListener("click", confirmDeleteAllCleaningDatasets);
        $("#cleaning-upload").addEventListener("change", event => cleaningUpload(event.target.files[0], false));
        $("#join-upload").addEventListener("change", event => cleaningUpload(event.target.files[0], true));
        $("#join-dataset-select").addEventListener("change", event => { cleaningState.second = cleaningState.datasets.find(item => item.id === event.target.value); renderJoinSchemas(); });
        $("#join-run").addEventListener("click", runJoin);
        $("#send-cleaned-comparison").addEventListener("click", sendCleanedToComparison);
        $("#validate-custom-cleaning").addEventListener("click", validateCustomCleaning);
        $("#save-custom-cleaning").addEventListener("click", saveCustomCleaning);
        await renderCustomCleaningActions();
        chooseCleaningOperation("remove_missing");
    }

    async function refreshCleaningDatasets(preferredId = null) {
        const response = await api("/api/datasets");
        cleaningState.datasets = response.datasets;
        cleaningState.active = cleaningState.datasets.find(item => item.id === (preferredId || response.active_id)) || cleaningState.datasets[0] || null;
        const options = cleaningState.datasets.map(item => `<option value="${item.id}" ${item.id === cleaningState.active?.id ? "selected" : ""}>${escapeHtml(item.filename)} · ${Number(item.rows).toLocaleString()} rows</option>`).join("");
        $("#cleaning-dataset-select").innerHTML = options || `<option value="">No datasets uploaded</option>`;
        $("#join-dataset-select").innerHTML = `<option value="">Choose a second uploaded dataset</option>` + cleaningState.datasets.filter(item => item.id !== cleaningState.active?.id).map(item => `<option value="${item.id}">${escapeHtml(item.filename)}</option>`).join("");
        renderCleaningSource(); renderCleaningFields();
    }

    function renderCleaningSource() {
        const data = cleaningState.active;
        $("#delete-cleaning-dataset").disabled = !data;
        const deleteAllButton = $("#delete-all-cleaning-datasets");
        if (deleteAllButton) deleteAllButton.disabled = !cleaningState.datasets.length;
        if (!data) {
            $("#cleaning-source-copy").textContent = "Upload a CSV or XLSX dataset to begin.";
            $("#cleaning-source-stats").innerHTML = "";
            renderTable($("#cleaning-preview"), []);
            $("#cleaning-preview-caption").textContent = "No dataset selected.";
            return;
        }
        $("#cleaning-source-copy").textContent = `${data.filename} remains unchanged while you create working copies.`;
        $("#cleaning-source-stats").innerHTML = [["Rows", data.rows], ["Columns", data.columns], ["Missing", data.missing_cells], ["Duplicates", data.duplicate_rows]].map(([label, value]) => `<div class="stat-item"><span>${label}</span><strong>${Number(value).toLocaleString()}</strong></div>`).join("");
        renderTable($("#cleaning-preview"), data.preview, data.column_names);
        $("#cleaning-preview-caption").textContent = `Previewing ${data.filename}.`;
    }

    function confirmDeleteAllCleaningDatasets() {
        const count = cleaningState.datasets.length;
        if (!count) return;
        openModal(`<h2 id="modal-title">Delete All ${count} Datasets?</h2><p>This permanently deletes every uploaded and cleaned dataset for this account. Existing comparison history will remain, but deleted datasets cannot be restored.</p><div class="button-row modal-footer-actions"><button class="button secondary" data-close-modal>Back</button><button class="button danger" id="confirm-delete-all-datasets">Delete All Datasets</button></div>`);
        $("#confirm-delete-all-datasets").addEventListener("click", async () => {
            try {
                const response = await api("/api/datasets", { method: "DELETE" });
                closeModal();
                resetCleaningOutput();
                cleaningState.second = null;
                await refreshCleaningDatasets();
                toast(`${response.deleted_count} dataset${response.deleted_count === 1 ? "" : "s"} deleted.`);
            } catch (error) { toast(error.message, "error"); }
        });
    }

    function confirmDeleteCleaningDataset() {
        const dataset = cleaningState.active;
        if (!dataset) return;
        openModal(`<h2 id="modal-title">Delete ${escapeHtml(dataset.filename)}?</h2><p>This permanently deletes the uploaded file and its local metadata. Existing history entries will remain, but they cannot restore this dataset.</p><div class="button-row modal-footer-actions"><button class="button secondary" data-close-modal>Back</button><button class="button danger" id="confirm-delete-dataset">Delete Dataset</button></div>`);
        $("#confirm-delete-dataset").addEventListener("click", async () => {
            try {
                const response = await api(`/api/datasets/${dataset.id}`, { method: "DELETE" });
                closeModal();
                cleaningState.output = null;
                $("#cleaning-output").classList.add("hidden");
                await refreshCleaningDatasets(response.active_dataset?.id || null);
                toast(`${dataset.filename} was permanently deleted.`, "warning");
            } catch (error) { toast(error.message, "error"); }
        });
    }

    async function cleaningUpload(file, asSecond) {
        try {
            toast("Uploading and profiling the dataset…", "warning");
            const response = await uploadFile(file);
            if (!asSecond) resetCleaningOutput();
            await refreshCleaningDatasets(asSecond ? cleaningState.active?.id : response.dataset.id);
            if (asSecond) {
                cleaningState.second = response.dataset;
                $("#join-dataset-select").value = response.dataset.id;
                renderJoinSchemas();
            }
            toast("Dataset ready.");
        } catch (error) { toast(error.message, "error"); }
    }

    const cleaningDescriptions = {
        remove_missing: ["Remove Missing Rows", "Choose columns that must contain a value."],
        fill_missing: ["Fill Missing Values", "Choose a statistically sensible replacement."],
        remove_duplicates: ["Remove Duplicate Rows", "Keep the first copy of every repeated row."],
        replace: ["Replace Values", "Change matching values in one selected column."],
        remove_outliers: ["Handle Outliers", "Remove numeric rows beyond the standard 1.5 × IQR range."],
        normalize: ["Normalize Numbers", "Place selected numeric attributes on a comparable scale."],
        drop_columns: ["Remove Columns", "Permanently exclude noise from the new working copy."],
        rename_columns: ["Rename Columns", "Apply clearer attribute names to the new copy."],
        convert_type: ["Convert Data Type", "Convert a column to integer, decimal, Boolean, category, date, date and time, or text."],
        custom_cleaning: ["Custom Cleaning", "Create, save, and run trusted local Python cleaning actions."],
        join: ["Join Two Datasets", "Combine two tables using matching key columns."],
    };

    function chooseCleaningOperation(operation) {
        if (cleaningState.operation !== operation) resetCleaningOutput();
        cleaningState.operation = operation;
        $$(".tool-button").forEach(button => button.classList.toggle("active", button.dataset.operation === operation));
        const [title, description] = cleaningDescriptions[operation];
        $("#cleaning-action-title").textContent = title; $("#cleaning-action-description").textContent = description;
        $("#join-card").classList.toggle("hidden", operation !== "join");
        $("#custom-cleaning-card").classList.toggle("hidden", operation !== "custom_cleaning");
        $(".action-card").classList.toggle("hidden", operation === "join" || operation === "custom_cleaning");
        renderCleaningFields();
    }

    async function validateCustomCleaning() {
        const output = $("#custom-cleaning-validation");
        output.className = "validation-output";
        output.textContent = "Checking Python and local privacy rules…";
        try {
            await api("/api/custom-cleaning/validate", { method: "POST", body: { code: $("#custom-cleaning-code").value } });
            output.classList.add("success");
            output.textContent = "Validation passed. clean_data(df) and allowed imports were found.";
            return true;
        } catch (error) {
            output.classList.add("error");
            output.textContent = (error.payload?.errors || [error.message]).map(item => `• ${item}`).join("\n");
            throw error;
        }
    }

    async function saveCustomCleaning() {
        try {
            await validateCustomCleaning();
            const payload = { name: $("#custom-cleaning-name").value, description: $("#custom-cleaning-description").value, code: $("#custom-cleaning-code").value };
            const url = cleaningState.editingCustomId ? `/api/custom-cleaning/${cleaningState.editingCustomId}` : "/api/custom-cleaning";
            const response = await api(url, { method: cleaningState.editingCustomId ? "PUT" : "POST", body: payload });
            toast(`${response.action.name} saved locally.`);
            cleaningState.editingCustomId = null;
            $("#save-custom-cleaning").textContent = "Save Action";
            $("#custom-cleaning-name").value = "";
            $("#custom-cleaning-description").value = "";
            await renderCustomCleaningActions();
        } catch (error) { if (!error.payload?.errors) toast(error.message, "error"); }
    }

    async function renderCustomCleaningActions() {
        const response = await api("/api/custom-cleaning");
        cleaningState.customActions = response.actions;
        if (!cleaningState.editingCustomId && !$("#custom-cleaning-name").value.trim()) {
            $("#custom-cleaning-name").value = response.suggested_name || "Custom Cleaning 1";
        }
        const root = $("#custom-cleaning-list");
        root.innerHTML = response.actions.length ? response.actions.map(action => `<article class="saved-cleaning-action"><div><strong>${escapeHtml(action.name)}</strong><p>${escapeHtml(action.description || "No description provided.")}</p></div><div class="button-row"><button class="button primary small" data-run-cleaning="${escapeHtml(action.id)}">Run</button><button class="button secondary small" data-edit-cleaning="${escapeHtml(action.id)}">Modify</button><button class="button danger small" data-delete-cleaning="${escapeHtml(action.id)}">Delete</button></div></article>`).join("") : `<div class="empty-card">No Custom Cleaning Actions Saved Yet.</div>`;
        $$('[data-run-cleaning]').forEach(button => button.addEventListener("click", () => runCustomCleaning(button.dataset.runCleaning)));
        $$('[data-edit-cleaning]').forEach(button => button.addEventListener("click", () => editCustomCleaning(button.dataset.editCleaning)));
        $$('[data-delete-cleaning]').forEach(button => button.addEventListener("click", () => deleteCustomCleaning(button.dataset.deleteCleaning)));
    }

    function editCustomCleaning(actionId) {
        const action = cleaningState.customActions.find(item => item.id === actionId);
        if (!action) return;
        cleaningState.editingCustomId = action.id;
        $("#custom-cleaning-name").value = action.name;
        $("#custom-cleaning-description").value = action.description || "";
        $("#custom-cleaning-code").value = action.code;
        $("#save-custom-cleaning").textContent = "Update Action";
        $("#custom-cleaning-validation").className = "validation-output";
        $("#custom-cleaning-validation").textContent = "Modified action is ready for validation.";
    }

    function deleteCustomCleaning(actionId) {
        const action = cleaningState.customActions.find(item => item.id === actionId);
        if (!action) return;
        openModal(`<h2 id="modal-title">Delete ${escapeHtml(action.name)}?</h2><p>This removes the saved action from Custom Cleaning.</p><div class="button-row modal-footer-actions"><button class="button secondary" data-close-modal>Back</button><button class="button danger" id="confirm-delete-cleaning">Delete</button></div>`);
        $("#confirm-delete-cleaning").addEventListener("click", async () => {
            try { await api(`/api/custom-cleaning/${actionId}`, { method: "DELETE" }); closeModal(); await renderCustomCleaningActions(); toast(`${action.name} deleted.`); }
            catch (error) { toast(error.message, "error"); }
        });
    }

    async function runCustomCleaning(actionId) {
        if (!cleaningState.active) return toast("Upload or select a dataset first.", "warning");
        try {
            toast("Running the trusted local cleaning action…", "warning");
            const response = await api(`/api/custom-cleaning/${actionId}/run`, { method: "POST" });
            setCleaningOutput(response.dataset);
            showCleaningOutput(response.dataset, response.message);
            await refreshCleaningDatasets(cleaningState.active.id);
        } catch (error) { toast(error.message, "error"); }
    }

    function columnOptions(filter = () => true) {
        return (cleaningState.active?.column_summaries || []).filter(filter).map(item => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`).join("");
    }

    function columnChecks(numericOnly = false) {
        const items = (cleaningState.active?.column_summaries || []).filter(item => !numericOnly || /int|float|number/i.test(item.dtype));
        return `<div class="column-checks">${items.map(item => `<label><input type="checkbox" name="columns" value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</label>`).join("")}</div>`;
    }

    function renderCleaningFields() {
        const root = $("#cleaning-fields");
        if (!root || !cleaningState.active) { if (root) root.innerHTML = `<p>Upload or select a dataset first.</p>`; renderCleaningVisualization(); return; }
        const operation = cleaningState.operation;
        if (["remove_missing", "drop_columns"].includes(operation)) root.innerHTML = `<div style="grid-column:1/-1"><label>Select columns</label>${columnChecks()}</div>`;
        else if (operation === "fill_missing") root.innerHTML = `<label>Method<select name="method"><option value="median">Median for numbers</option><option value="mean">Mean for numbers</option><option value="mode">Most common value</option><option value="constant">Custom constant</option></select></label><label>Constant value<input name="value" placeholder="Used for custom constant"></label><div style="grid-column:1/-1">${columnChecks()}</div>`;
        else if (operation === "remove_duplicates") root.innerHTML = `<p>This checks complete rows. The original dataset is preserved.</p>`;
        else if (operation === "replace") root.innerHTML = `<label>Column<select name="column">${columnOptions()}</select></label><label>Find Value<input name="find"></label><label>Replace With<input name="replace"></label><label class="condition-field">Condition (Optional)<input name="condition" placeholder="e.g. ID < 30 OR ID > 1000"><small>Use AND, OR, parentheses, and =, !=, &lt;, &lt;=, &gt;, or &gt;=. Put column names with spaces inside [brackets] and text inside quotes.</small></label>`;
        else if (operation === "remove_outliers") root.innerHTML = `<div style="grid-column:1/-1"><label>Numeric columns</label>${columnChecks(true)}</div>`;
        else if (operation === "normalize") root.innerHTML = `<label>Scaling method<select name="method"><option value="standard">Standard score (mean 0)</option><option value="minmax">Min-max (0 to 1)</option></select></label><div style="grid-column:1/-1">${columnChecks(true)}</div>`;
        else if (operation === "rename_columns") root.innerHTML = `<label>Existing column<select name="rename_source">${columnOptions()}</select></label><label>New name<input name="rename_target" required></label>`;
        else if (operation === "convert_type") root.innerHTML = `<label>Column<select name="column">${columnOptions()}</select></label><label>Data Type<select name="dtype"><option value="integer">Integer</option><option value="decimal">Decimal</option><option value="boolean">Boolean</option><option value="category">Category</option><option value="date">Date</option><option value="datetime">Date and Time</option><option value="text">Text</option></select></label>`;
        renderCleaningVisualization();
    }

    function resetCleaningOutput() {
        cleaningState.output = null;
        cleaningState.outputOperation = null;
        cleaningState.outputSourceId = null;
        $("#cleaning-output")?.classList.add("hidden");
    }

    function setCleaningOutput(dataset) {
        cleaningState.output = dataset;
        cleaningState.outputOperation = cleaningState.operation;
        cleaningState.outputSourceId = cleaningState.active?.id || null;
    }

    function renderCleaningVisualization() {
        const root = $("#cleaning-visualization");
        if (!root) return;
        const hasCurrentOutput = cleaningState.output
            && cleaningState.outputOperation === cleaningState.operation
            && cleaningState.outputSourceId === cleaningState.active?.id;
        const dataset = hasCurrentOutput ? cleaningState.output : cleaningState.active;
        root.classList.toggle("hidden", !dataset);
        if (!dataset) { root.innerHTML = ""; return; }

        const stage = hasCurrentOutput ? "Cleaned Dataset" : "Original Dataset";
        const operationName = cleaningDescriptions[cleaningState.operation]?.[0] || "Cleaning";
        const heading = `<div class="visualization-heading"><span class="visualization-stage ${hasCurrentOutput ? "cleaned" : "original"}">${stage}</span><h3>${escapeHtml(operationName)} Visualization</h3><p>${hasCurrentOutput ? `Updated using ${escapeHtml(dataset.filename)}.` : "This preview updates to the cleaned result after the action finishes."}</p></div>`;

        if (["remove_missing", "fill_missing"].includes(cleaningState.operation)) {
            const columns = dataset.column_summaries || [];
            const maximum = Math.max(...columns.map(column => Number(column.missing || 0)), 1);
            root.innerHTML = `${heading}<div class="missing-chart">${columns.map(column => { const missing = Number(column.missing || 0); return `<div class="missing-chart-row"><span title="${escapeHtml(column.name)}">${escapeHtml(column.name)}</span><div class="missing-chart-track"><div class="missing-chart-bar" style="width:${missing / maximum * 100}%"></div></div><strong>${missing.toLocaleString()}</strong></div>`; }).join("")}</div>`;
            return;
        }

        const rows = Number(dataset.rows || 0);
        const columns = Number(dataset.columns || 0);
        const missing = Number(dataset.missing_cells || 0);
        const duplicates = Number(dataset.duplicate_rows || 0);
        const totalCells = Math.max(rows * columns, 1);
        const originalRows = Math.max(Number(cleaningState.active?.rows || rows), 1);
        const originalColumns = Math.max(Number(cleaningState.active?.columns || columns), 1);
        const metrics = [
            ["Complete Cells", Math.max(0, 100 - missing / totalCells * 100), `${Math.max(0, totalCells - missing).toLocaleString()} / ${totalCells.toLocaleString()}`],
            ["Unique Rows", rows ? Math.max(0, 100 - duplicates / rows * 100) : 100, `${Math.max(0, rows - duplicates).toLocaleString()} / ${rows.toLocaleString()}`],
            ["Rows Retained", Math.min(100, rows / originalRows * 100), rows.toLocaleString()],
            ["Columns Retained", Math.min(100, columns / originalColumns * 100), columns.toLocaleString()],
        ];
        root.innerHTML = `${heading}<div class="quality-chart">${metrics.map(([label, percent, value]) => `<div class="quality-chart-item"><div><span>${label}</span><strong>${value}</strong></div><div class="quality-chart-track"><span style="width:${percent}%"></span></div></div>`).join("")}</div>`;
    }

    async function runCleaning(event) {
        event.preventDefault();
        if (!cleaningState.active) return toast("Upload or select a dataset first.", "warning");
        const form = new FormData(event.currentTarget);
        const options = Object.fromEntries(form.entries());
        options.columns = form.getAll("columns");
        if (cleaningState.operation === "rename_columns") options.mapping = { [options.rename_source]: options.rename_target };
        try {
            toast("Creating a cleaned working copy…", "warning");
            const response = await api("/api/cleaning/apply", { method: "POST", body: { operation: cleaningState.operation, options } });
            setCleaningOutput(response.dataset);
            showCleaningOutput(response.dataset, response.message);
            await refreshCleaningDatasets(cleaningState.active.id);
        } catch (error) { toast(error.message, "error"); }
    }

    function renderJoinSchemas() {
        const left = cleaningState.active, right = cleaningState.second;
        const schema = (data, side) => (data?.column_summaries || []).map(item => `<div class="schema-item editable"><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.dtype)}</small></span><input class="join-rename" data-side="${side}" data-column="${escapeHtml(item.name)}" value="${escapeHtml(item.name)}" aria-label="Rename ${escapeHtml(item.name)} from ${side} dataset"></div>`).join("") || `<div class="schema-item"><span>No dataset selected</span></div>`;
        $("#left-schema").innerHTML = schema(left, "left"); $("#right-schema").innerHTML = schema(right, "right");
        $("#left-join-key").innerHTML = (left?.column_names || []).map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
        $("#right-join-key").innerHTML = (right?.column_names || []).map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    }

    async function runJoin() {
        if (!cleaningState.active || !cleaningState.second) return toast("Choose both datasets first.", "warning");
        const keep = $$("#duplicate-options input:checked").map(input => input.value);
        const leftRename = {}, rightRename = {};
        $$('.join-rename[data-side="left"]').forEach(input => { if (input.value.trim() && input.value.trim() !== input.dataset.column) leftRename[input.dataset.column] = input.value.trim(); });
        $$('.join-rename[data-side="right"]').forEach(input => { if (input.value.trim() && input.value.trim() !== input.dataset.column) rightRename[input.dataset.column] = input.value.trim(); });
        try {
            const response = await api("/api/cleaning/join", { method: "POST", body: {
                right_dataset_id: cleaningState.second.id, left_key: $("#left-join-key").value,
                right_key: $("#right-join-key").value, how: $("#join-type").value,
                left_suffix: "_left", right_suffix: $("#right-suffix").value || "_right",
                keep_duplicate_columns: keep, left_rename: leftRename, right_rename: rightRename,
            }});
            setCleaningOutput(response.dataset);
            cleaningState.removedDuplicates = response.removed_duplicate_columns;
            if (response.removed_duplicate_columns.length) {
                $("#duplicate-restore").classList.remove("hidden");
                $("#duplicate-options").innerHTML = response.removed_duplicate_columns.map(item => `<label><input type="checkbox" value="${escapeHtml(item.column)}">${escapeHtml(item.column)}</label>`).join("");
            } else $("#duplicate-restore").classList.add("hidden");
            showCleaningOutput(response.dataset, response.message);
            await refreshCleaningDatasets(cleaningState.active.id);
        } catch (error) { toast(error.message, "error"); }
    }

    function showCleaningOutput(dataset, message) {
        renderCleaningVisualization();
        const output = $("#cleaning-output"); output.classList.remove("hidden");
        $("#cleaning-result-message").textContent = `${message} New copy: ${dataset.filename} (${Number(dataset.rows).toLocaleString()} rows × ${dataset.columns} columns).`;
        $("#download-cleaned-csv").href = `/api/datasets/${dataset.id}/download.csv`;
        $("#download-cleaned-xlsx").href = `/api/datasets/${dataset.id}/download.xlsx`;
        output.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    async function sendCleanedToComparison() {
        if (!cleaningState.output) return;
        openModal(`<h2 id="modal-title">Move This Dataset to Comparison?</h2><p><strong>${escapeHtml(cleaningState.output.filename)}</strong> will become the active imported dataset. Your original remains available.</p><div class="button-row"><button class="button primary" id="confirm-send-cleaned">Yes, Open Comparison</button><button class="button secondary" data-close-modal>Stay Here</button></div>`);
        $("#confirm-send-cleaned").addEventListener("click", async () => {
            const response = await api(`/api/datasets/${cleaningState.output.id}/send-to-comparison`, { method: "POST" });
            location.href = response.redirect;
        });
    }

    // --------------------
    // Supporting Pages
    // Handle the model library, saved history, custom student code and website settings.
    // --------------------
    async function initLibraries() {
        await loadSession();
        const response = await api("/api/models"); state.models = response.models;
        populateLibraryTypes();
        $("#library-search").addEventListener("input", renderLibrary);
        $("#library-task").addEventListener("change", renderLibrary);
        renderLibrary();
    }

    function populateLibraryTypes() {
        const select = $("#library-task");
        if (!select) return;
        const customTags = [...new Set(state.models.filter(model => model.id.startsWith("custom:")).map(model => model.task_label).filter(Boolean))]
            .sort((a, b) => a.localeCompare(b));
        select.innerHTML = `<option value="all">All Model Types</option><option value="classification">Classification</option><option value="regression">Regression</option><option value="clustering">Clustering</option><option value="custom">Custom Model</option>${customTags.map(tag => `<option value="tag:${escapeHtml(tag)}">${escapeHtml(tag)}</option>`).join("")}`;
    }

    function renderLibrary() {
        const query = ($("#library-search")?.value || "").toLowerCase();
        const task = $("#library-task")?.value || "all";
        const items = state.models.filter(model => {
            const isCustom = model.id.startsWith("custom:");
            const matchesQuery = !query || [model.name, model.family, model.summary, model.best_for, model.task_label, isCustom ? "custom model" : ""].join(" ").toLowerCase().includes(query);
            const matchesType = task === "all" || (task === "custom" && isCustom) || (task.startsWith("tag:") && isCustom && model.task_label === task.slice(4)) || model.tasks.includes(task);
            return matchesQuery && matchesType;
        });
        $("#library-grid").innerHTML = items.map(model => { const isCustom = model.id.startsWith("custom:"); const tags = isCustom ? `<span class="task-tag user-model-tag">Custom Model</span>${model.task_label ? `<span class="task-tag">${escapeHtml(model.task_label)}</span>` : ""}` : model.tasks.map(item => `<span class="task-tag">${escapeHtml(titleCase(item))}</span>`).join(""); return `<article class="card library-card"><div class="library-card-head"><div><span class="family">${escapeHtml(model.family)}</span><h2>${escapeHtml(model.name)}</h2></div></div><div class="task-tags">${tags}</div><p>${escapeHtml(model.summary)}</p><div class="best-box" tabindex="0" title="Scroll to read the full use case"><strong>Best Used When</strong><br>${escapeHtml(model.best_for)}</div><div class="library-card-actions"><button class="button secondary small use-library-model" data-library-model="${escapeHtml(model.id)}">Use in Comparison</button>${isCustom ? `<a class="button ghost small" href="/custom-models#saved-models">Manage Model</a>` : ""}</div></article>`; }).join("");
        $$('.use-library-model').forEach(button => button.addEventListener("click", () => { localStorage.setItem("pendingModel", button.dataset.libraryModel); location.href = "/comparison"; }));
    }

    async function initHistory() {
        await loadSession();
        await renderHistory();
        $("#clear-history").addEventListener("click", () => {
            openModal(`<h2 id="modal-title">Delete All Comparison History?</h2><p>This removes the local history list for your account. Exported files are not affected.</p><button class="button danger" id="confirm-clear-history">Delete Local History</button>`);
            $("#confirm-clear-history").addEventListener("click", async () => { await api("/api/history", { method: "DELETE" }); closeModal(); renderHistory(); toast("Local history deleted."); });
        });
    }

    async function renderHistory() {
        const response = await api("/api/history");
        const root = $("#history-list");
        if (!response.history.length) { root.innerHTML = `<div class="empty-card history-empty"><strong>No Comparisons Yet</strong><br>Completed comparisons will appear here automatically.</div>`; return; }
        root.innerHTML = response.history.map(item => `<article class="card history-card"><div><h2>${item.models.map(model => escapeHtml(model.name)).join(" <span class='muted'>vs</span> ")}</h2><div class="history-meta"><span>${escapeHtml(item.dataset_name)}</span><span>Target: ${escapeHtml(item.target)}</span><span>${escapeHtml(titleCase(item.task))}</span><span>${item.train_pct}/${100 - item.train_pct} split</span><span>${escapeHtml(item.mode_label || titleCase(item.mode || "full"))} budget</span><span>${Number(item.rows_processed || 0).toLocaleString()} / ${Number(item.rows_available || item.rows_processed || 0).toLocaleString()} rows</span><span>${new Date(item.created_at).toLocaleString()}</span></div><div class="history-models">${item.models.map(model => `<span class="history-model"><strong>${escapeHtml(model.name)}</strong> · ${Object.entries(model.metrics).slice(0, 2).map(([key, value]) => `${escapeHtml(titleCase(key))}: ${escapeHtml(formatMetric(key, value))}`).join(" · ")}</span>`).join("")}</div></div><div class="button-row"><button class="button secondary small history-view-button view-history-result" data-result-id="${item.id}">View</button></div></article>`).join("");
        $$('.view-history-result').forEach(button => button.addEventListener("click", () => viewHistoryResult(button.dataset.resultId)));
    }

    async function viewHistoryResult(resultId) {
        try {
            const response = await api(`/api/results/${resultId}`);
            const result = response.result;
            const receipt = result.affordability || {};
            openModal(`<h2 id="modal-title">${result.models.map(model => escapeHtml(model.name)).join(" vs ")}</h2><p>${escapeHtml(result.dataset_name)} · target ${escapeHtml(result.target)} · ${result.train_pct}/${result.test_pct} split</p><div class="history-receipt"><strong>Affordability receipt</strong><span>RM${Number(receipt.application_fee_myr || 0).toFixed(2)} application fee · ${Number(receipt.paid_external_api_calls || 0)} paid ML API calls · ${escapeHtml(receipt.processing_location || "Local computer")}</span><span>${escapeHtml(receipt.resource_mode_label || titleCase(result.mode || "full"))} · ${Number(result.rows_processed).toLocaleString()} / ${Number(result.rows_available || result.rows_processed).toLocaleString()} rows · ${formatNumber(receipt.training_seconds ?? result.total_training_seconds ?? 0)} s training</span></div><div class="modal-options">${result.models.map(model => `<div class="modal-option"><span><strong>${escapeHtml(model.name)}</strong><small>${Object.entries(model.metrics).map(([key,value]) => `${escapeHtml(titleCase(key))}: ${escapeHtml(formatMetric(key,value))}`).join(" · ")}</small></span></div>`).join("")}</div><div class="button-row modal-footer-actions"><button class="button primary" id="restore-history-comparison">Show in Comparison</button></div>`);
            $("#restore-history-comparison").addEventListener("click", async () => {
                try {
                    const restored = await api(`/api/history/${result.id}/restore`, { method: "POST" });
                    localStorage.setItem("restoreComparison", JSON.stringify(restored.comparison));
                    location.href = restored.redirect;
                } catch (error) { toast(error.message, "error"); }
            });
        } catch (error) { toast(error.message, "error"); }
    }

    const customState = { editingId: null, models: [], parameters: [{ name: "max_depth", type: "integer", default: 5, min: 1, max: 100, description: "Maximum number of tree levels" }] };

    async function initCustomModels() {
        await loadSession(); renderParameters();
        $("#custom-name").value = "";
        await renderSavedCustom();
        $("#add-parameter").addEventListener("click", () => { customState.parameters.push({ name: "parameter", type: "number", default: 1, description: "Explain what this changes" }); renderParameters(); });
        $("#validate-code").addEventListener("click", validateCode);
        $("#save-custom-model").addEventListener("click", saveCustomModel);
        $("#custom-model-tag").addEventListener("input", syncCustomModelTag);
        $("#custom-name").addEventListener("input", updateCustomNameCount);
        updateCustomNameCount();
        syncCustomModelTag();
    }

    function inferCustomTask(tag) {
        const value = String(tag || "").toLowerCase();
        if (/cluster|segment|group/.test(value)) return "clustering";
        if (/regress|regular|numeric|continuous|number/.test(value)) return "regression";
        return "classification";
    }

    function syncCustomModelTag() {
        const task = inferCustomTask($("#custom-model-tag")?.value);
        if ($("#custom-task")) $("#custom-task").value = task;
    }

    function updateCustomNameCount() {
        const words = ($("#custom-name")?.value.trim().match(/\S+/g) || []).length;
        const count = $("#custom-name-count");
        if (count) { count.textContent = `${words} / 30 words`; count.classList.toggle("error-text", words > 30); }
        return words;
    }

    function renderParameters() {
        $("#parameter-list").innerHTML = customState.parameters.map((parameter, index) => { const range = parameter.range ?? ((parameter.min !== undefined || parameter.max !== undefined) ? `${parameter.min ?? ""}..${parameter.max ?? ""}` : "none"); return `<div class="parameter-row" data-param-index="${index}"><div class="parameter-row-grid"><label>Title<input data-field="name" value="${escapeHtml(parameter.name)}" placeholder="e.g. max_depth"></label><label>Data Type<select data-field="type">${["integer", "number", "boolean", "text", "choice"].map(type => `<option value="${type}" ${type === parameter.type ? "selected" : ""}>${titleCase(type)}</option>`).join("")}</select></label><label>Range or None<input data-field="range" value="${escapeHtml(range)}" placeholder="1..100 or none"></label></div><details><summary>Default Value and Description</summary><div class="parameter-range-fields"><input data-field="default" value="${escapeHtml(parameter.default)}" placeholder="Default value"><input data-field="description" value="${escapeHtml(parameter.description || "")}" placeholder="Simple description"></div></details><div class="parameter-row-actions"><button class="button ghost small" data-remove-parameter="${index}">Remove Parameter</button></div></div>`; }).join("");
        $$("#parameter-list [data-field]").forEach(input => input.addEventListener("input", () => { customState.parameters[Number(input.closest(".parameter-row").dataset.paramIndex)][input.dataset.field] = input.value; }));
        $$('[data-remove-parameter]').forEach(button => button.addEventListener("click", () => { customState.parameters.splice(Number(button.dataset.removeParameter), 1); renderParameters(); }));
    }

    async function validateCode() {
        const output = $("#validation-output"); output.className = "validation-output"; output.textContent = "Checking syntax and privacy rules…";
        try {
            const response = await api("/api/custom-models/validate", { method: "POST", body: { code: $("#custom-code").value } });
            output.classList.add("success"); output.textContent = "Validation passed. The required function and allowed imports were found.";
            return response;
        } catch (error) {
            output.classList.add("error"); output.textContent = (error.payload?.errors || [error.message]).map(item => `• ${item}`).join("\n");
            throw error;
        }
    }

    async function saveCustomModel() {
        try {
            if (updateCustomNameCount() > 30) throw new Error("Model Name must contain no more than 30 words.");
            await validateCode();
            const parameters = customState.parameters.map(parameter => {
                const item = { ...parameter };
                delete item.min; delete item.max;
                const range = String(item.range || "").trim();
                if (["integer", "number"].includes(item.type) && range && range.toLowerCase() !== "none") {
                    const [minimum, maximum] = range.split("..");
                    if (minimum?.trim()) item.min = minimum.trim();
                    if (maximum?.trim()) item.max = maximum.trim();
                }
                delete item.range;
                return item;
            });
            const url = customState.editingId ? `/api/custom-models/${customState.editingId}` : "/api/custom-models";
            const response = await api(url, { method: customState.editingId ? "PUT" : "POST", body: {
                name: $("#custom-name").value, task: $("#custom-task").value,
                task_label: $("#custom-model-tag").value.trim() || titleCase($("#custom-task").value),
                description: $("#custom-description").value, code: $("#custom-code").value,
                parameters,
            }});
            toast(`${response.model.name} ${customState.editingId ? "updated" : "saved to your local model shelf"}.`);
            customState.editingId = null;
            $("#save-custom-model").textContent = "Save to Model Shelf";
            $("#custom-name").value = "";
            updateCustomNameCount();
            await renderSavedCustom();
        } catch (error) { if (!error.payload?.errors) toast(error.message, "error"); }
    }

    async function renderSavedCustom() {
        const response = await api("/api/custom-models");
        customState.models = response.models;
        if (!customState.editingId && !$("#custom-name").value.trim()) {
            $("#custom-name").value = response.suggested_name || "Custom Model 1";
            updateCustomNameCount();
        }
        const suggestions = $("#model-tag-suggestions");
        if (suggestions) {
            const defaults = ["Classification", "Regression", "Regularised Regression", "Clustering", "Recommendation", "Anomaly Detection"];
            const tags = [...new Set([...defaults, ...response.models.map(model => model.task_label).filter(Boolean)])];
            suggestions.innerHTML = tags.map(tag => `<option value="${escapeHtml(tag)}"></option>`).join("");
        }
        $("#custom-model-list").innerHTML = response.models.length ? response.models.map(model => `<article class="card library-card"><span class="family">Trusted Local · Custom Model</span><h2>${escapeHtml(model.name)}</h2><div class="task-tags"><span class="task-tag user-model-tag">Custom Model</span>${model.task_label ? `<span class="task-tag">${escapeHtml(model.task_label)}</span>` : ""}</div><p>${escapeHtml(model.description)}</p><div class="task-tags">${model.parameters.map(parameter => `<span class="task-tag">${escapeHtml(parameter.name)} = ${escapeHtml(parameter.default)}</span>`).join("")}</div><div class="library-card-actions"><button class="button secondary small" data-edit-custom="${escapeHtml(model.id)}">Modify Model</button><button class="button danger small" data-delete-custom="${escapeHtml(model.id)}">Delete Model</button></div></article>`).join("") : `<div class="empty-card">No Custom Models Saved Yet.</div>`;
        $$('[data-edit-custom]').forEach(button => button.addEventListener("click", () => editCustomModel(button.dataset.editCustom)));
        $$('[data-delete-custom]').forEach(button => button.addEventListener("click", () => confirmDeleteCustomModel(button.dataset.deleteCustom)));
    }

    function editCustomModel(modelId) {
        const model = customState.models.find(item => item.id === modelId);
        if (!model) return;
        customState.editingId = model.id;
        customState.parameters = model.parameters.map(parameter => ({ ...parameter }));
        $("#custom-name").value = model.name;
        updateCustomNameCount();
        $("#custom-task").value = model.task;
        $("#custom-model-tag").value = model.task_label || titleCase(model.task);
        syncCustomModelTag();
        $("#custom-description").value = model.description || "";
        $("#custom-code").value = model.code;
        $("#save-custom-model").textContent = "Update Model";
        $("#validation-output").className = "validation-output";
        $("#validation-output").textContent = "Modified model is ready for validation.";
        renderParameters();
        $(".main-area")?.scrollTo({ top: 0, behavior: "smooth" });
    }

    function confirmDeleteCustomModel(modelId) {
        const model = customState.models.find(item => item.id === modelId);
        if (!model) return;
        openModal(`<h2 id="modal-title">Delete ${escapeHtml(model.name)}?</h2><p>This removes the model definition from this local account and the Model Shelf.</p><button class="button danger" id="confirm-delete-custom">Delete Model</button>`);
        $("#confirm-delete-custom").addEventListener("click", async () => {
            try {
                await api(`/api/custom-models/${modelId}`, { method: "DELETE" });
                if (customState.editingId === modelId) customState.editingId = null;
                closeModal(); await renderSavedCustom(); toast(`${model.name} deleted.`);
            } catch (error) { toast(error.message, "error"); }
        });
    }

    async function initSettings() {
        const response = await loadSession();
        const settingsResponse = await api("/api/settings"); state.settings = settingsResponse.settings;
        $("#setting-theme").value = state.settings.theme;
        $("#setting-tooltips").checked = state.settings.tooltips;
        $("#setting-train").value = state.settings.train_pct;
        $("#setting-seed").value = state.settings.random_seed;
        $("#setting-processing").value = state.settings.processing;
        $("#setting-export").value = state.settings.export;
        $("#profile-name").value = response.user.display_name || response.user.username;
        const profileEmail = $("#profile-email");
        if (profileEmail) profileEmail.value = response.user.email || "";
        $("#save-settings").addEventListener("click", saveSettings);
        $("#save-profile").addEventListener("click", saveProfile);
    }

    async function saveSettings() {
        try {
            const response = await api("/api/settings", { method: "PUT", body: {
                theme: $("#setting-theme").value, tooltips: $("#setting-tooltips").checked,
                train_pct: Number($("#setting-train").value), random_seed: Number($("#setting-seed").value),
                processing: $("#setting-processing").value, export: $("#setting-export").value,
            }});
            state.settings = response.settings; document.documentElement.dataset.theme = state.settings.theme;
            document.documentElement.dataset.tooltips = state.settings.tooltips ? "on" : "off";
            toast("Preferences saved locally.");
        } catch (error) { toast(error.message, "error"); }
    }

    async function saveProfile() {
        try {
            const response = await api("/api/account/profile", { method: "POST", body: {
                display_name: $("#profile-name").value,
                email: $("#profile-email")?.value || state.session?.user?.email || "",
                current_password: $("#current-password").value,
                new_password: $("#new-password").value,
            }});
            const displayName = response.user.display_name || response.user.username;
            const sidebarName = $(".user-copy strong");
            const avatar = $(".user-chip .avatar");
            if (sidebarName) sidebarName.textContent = displayName;
            if (avatar) avatar.textContent = displayName.charAt(0).toUpperCase();
            $("#current-password").value = ""; $("#new-password").value = ""; toast("Local account updated.");
        } catch (error) { toast(error.message, "error"); }
    }

    async function start() {
        bindGlobalUI();
        try {
            if (page === "comparison") await initComparison();
            else if (page === "cleaning") await initCleaning();
            else if (page === "libraries") await initLibraries();
            else if (page === "history") await initHistory();
            else if (page === "custom_models") await initCustomModels();
            else if (page === "settings") await initSettings();
        } catch (error) {
            toast(error.message, "error", 7000);
            console.error(error);
        }
    }

    document.addEventListener("DOMContentLoaded", start);
})();

