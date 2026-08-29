document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. Tab Navigation & View Swapping
    // -------------------------------------------------------------
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute("data-tab");

            // Toggle active classes
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");

            tabContents.forEach(content => {
                content.classList.remove("active");
                if (content.getAttribute("id") === targetTab) {
                    content.classList.add("active");
                }
            });

            // Trigger specific loads
            if (targetTab === "data-profiler") {
                loadDataProfiler();
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Multi-Theme Skin Customizer
    // -------------------------------------------------------------
    const themeButtons = document.querySelectorAll(".theme-btn");
    themeButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const selectedTheme = btn.getAttribute("data-theme");
            
            // Remove active style from all
            themeButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Apply selected theme class to body
            document.body.className = selectedTheme;
        });
    });

    // -------------------------------------------------------------
    // 3. Globals & Metrics Dashboard Loading
    // -------------------------------------------------------------
    let featureChartInstance = null;
    let xaiChartInstance = null;
    let compareChartInstance = null;
    let batchChartInstance = null;
    let profileContractChartInstance = null;
    let profileInternetChartInstance = null;

    async function loadMetrics() {
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();

            // Update Dashboard KPI cards
            if (data.metrics) {
                document.getElementById("kpi-auc").innerText = (data.metrics.roc_auc || 0).toFixed(4);
                document.getElementById("kpi-accuracy").innerText = ((data.metrics.accuracy || 0) * 100).toFixed(2) + "%";
                document.getElementById("kpi-precision").innerText = ((data.metrics.precision || 0) * 100).toFixed(2) + "%";
                document.getElementById("kpi-f1").innerText = ((data.metrics.f1_score || 0) * 100).toFixed(2) + "%";

                // Update confusion matrix cells
                const cm = data.metrics.confusion_matrix;
                if (cm && cm.length === 2) {
                    document.getElementById("cm-tn").innerText = cm[0][0]; // True Negative
                    document.getElementById("cm-fp").innerText = cm[0][1]; // False Positive
                    document.getElementById("cm-fn").innerText = cm[1][0]; // False Negative
                    document.getElementById("cm-tp").innerText = cm[1][1]; // True Positive
                    
                    // Calc matrix rates
                    const totalNeg = cm[0][0] + cm[0][1];
                    const totalPos = cm[1][0] + cm[1][1];
                    const spec = totalNeg > 0 ? (cm[0][0] / totalNeg * 100).toFixed(1) : "0.0";
                    const rec = totalPos > 0 ? (cm[1][1] / totalPos * 100).toFixed(1) : "0.0";
                    document.getElementById("cm-stats-lbl").innerText = `Specificity: ~${spec}% | Sensitivity (Recall): ~${rec}%`;
                }
            }

            // Update Model Registry details
            if (data.model_report) {
                document.getElementById("best-model-lbl").innerText = data.model_report.best_model || "None";
                
                // Build comparison rows
                const tbody = document.getElementById("model-comparison-rows");
                if (tbody && data.model_report.all_models) {
                    tbody.innerHTML = "";
                    Object.entries(data.model_report.all_models).forEach(([name, details]) => {
                        const isActive = data.model_report.best_model === name;
                        const row = document.createElement("tr");
                        row.innerHTML = `
                            <td>
                                <strong>${name}</strong> 
                                ${isActive ? '<span class="badge badge-success" style="font-size: 9px; padding: 2px 6px; margin-left: 6px;">Active</span>' : ''}
                            </td>
                            <td style="font-weight: 700; color: white;">${(details.roc_auc || 0).toFixed(4)}</td>
                            <td><code style="font-size: 11px; color: var(--accent);">${JSON.stringify(details.best_params)}</code></td>
                            <td>
                                <button class="btn btn-secondary btn-promote-sandbox" data-model="${name}" style="padding: 6px 12px; font-size: 11px;" ${isActive ? 'disabled' : ''}>
                                    <i class="fa-solid fa-cloud-arrow-up"></i> Promote Staging
                                </button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });

                    // Add promote listeners
                    document.querySelectorAll(".btn-promote-sandbox").forEach(btn => {
                        btn.addEventListener("click", async () => {
                            const modelName = btn.getAttribute("data-model");
                            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Promoting...';
                            await promoteModel(modelName);
                        });
                    });
                }
            }

            // Great Expectations schema assertions
            if (data.ge_validation) {
                const geList = document.getElementById("ge-assertions-list");
                if (geList) {
                    geList.innerHTML = "";
                    data.ge_validation.forEach(assertion => {
                        const div = document.createElement("div");
                        div.className = "ge-item";
                        
                        let badgeClass = "badge-success";
                        let statusIcon = '<i class="fa-solid fa-circle-check" style="color: var(--success)"></i>';
                        if (assertion.status === "FAILED") {
                            badgeClass = "badge-danger";
                            statusIcon = '<i class="fa-solid fa-triangle-exclamation" style="color: var(--danger)"></i>';
                        } else if (assertion.status === "WARNING") {
                            badgeClass = "badge-warning";
                            statusIcon = '<i class="fa-solid fa-triangle-exclamation" style="color: var(--warning)"></i>';
                        }

                        div.innerHTML = `
                            <div>
                                <span class="ge-assertion">${assertion.assertion}()</span>
                                <div class="ge-details">${assertion.details}</div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="badge ${badgeClass}">${assertion.status}</span>
                                ${statusIcon}
                            </div>
                        `;
                        geList.appendChild(div);
                    });
                }
            }
        } catch (err) {
            console.error("Error loading metrics:", err);
        }
    }

    // Model Promotion trigger
    async function promoteModel(modelName) {
        try {
            const res = await fetch("/api/promote-model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: modelName })
            });
            const data = await res.json();
            if (data.success) {
                await loadMetrics();
                loadFeaturesChart();
                // Trigger sandbox update in case form is modified
                triggerSandboxPrediction();
            } else {
                alert("Model promotion failed: " + data.message);
            }
        } catch (err) {
            console.error("Error promoting model:", err);
        }
    }

    // Feature Importances bar chart loader
    async function loadFeaturesChart() {
        try {
            const res = await fetch("/api/features");
            const data = await res.json();

            const canvas = document.getElementById("featureChart");
            const errDiv = document.getElementById("feature-chart-error");
            if (!canvas) return;

            if (!data.success) {
                errDiv.innerText = data.message;
                return;
            } else {
                errDiv.innerText = "";
            }

            const labels = data.features.map(f => f.name);
            const values = data.features.map(f => f.importance);

            if (featureChartInstance) {
                featureChartInstance.destroy();
            }

            const ctx = canvas.getContext("2d");
            featureChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Relative Importance Score',
                        data: values,
                        backgroundColor: 'rgba(6, 182, 212, 0.7)',
                        borderColor: 'rgba(6, 182, 212, 1)',
                        borderWidth: 1,
                        borderRadius: 5
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.03)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    }
                }
            });
        } catch (err) {
            console.error("Error drawing feature chart:", err);
        }
    }

    // Initial system load
    loadMetrics();
    loadFeaturesChart();

    // -------------------------------------------------------------
    // 4. Data Profiler Tab & Stats Loader
    // -------------------------------------------------------------
    async function loadDataProfiler() {
        try {
            const res = await fetch("/api/data-profile");
            const data = await res.json();
            if (!data.success) return;

            // Render numeric stats summary cards
            const statsContainer = document.getElementById("profiler-stats-container");
            if (statsContainer) {
                statsContainer.innerHTML = "";
                Object.entries(data.num_stats).forEach(([col, stats]) => {
                    const card = document.createElement("div");
                    card.className = "profile-stat-card";
                    card.innerHTML = `
                        <div class="profile-stat-title">${col}</div>
                        <div class="profile-stat-rows">
                            <div class="profile-stat-row">
                                <span>Mean / Median</span>
                                <span>${stats.mean} / ${stats.median}</span>
                            </div>
                            <div class="profile-stat-row">
                                <span>Standard Dev</span>
                                <span>${stats.std}</span>
                            </div>
                            <div class="profile-stat-row">
                                <span>Min / Max Bounds</span>
                                <span>${stats.min} / ${stats.max}</span>
                            </div>
                            <div class="profile-stat-row" style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px; margin-top: 4px;">
                                <span>Missing Values</span>
                                <span style="color: ${stats.missing_ratio > 0 ? 'var(--warning)' : 'var(--success)'};">${stats.missing_ratio}%</span>
                            </div>
                        </div>
                    `;
                    statsContainer.appendChild(card);
                });
            }

            // Draw Categorical splits
            const contractCounts = data.cat_distributions.Contract;
            const internetCounts = data.cat_distributions.InternetService;

            // Contract Distribution Chart
            if (profileContractChartInstance) profileContractChartInstance.destroy();
            const contractCtx = document.getElementById("profileContractChart").getContext("2d");
            profileContractChartInstance = new Chart(contractCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(contractCounts),
                    datasets: [{
                        data: Object.values(contractCounts),
                        backgroundColor: ['#2563eb', '#06b6d4', '#10b981'],
                        borderColor: '#060913',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#94a3b8', font: { size: 11 } }
                        }
                    }
                }
            });

            // Internet Distribution Chart
            if (profileInternetChartInstance) profileInternetChartInstance.destroy();
            const internetCtx = document.getElementById("profileInternetChart").getContext("2d");
            profileInternetChartInstance = new Chart(internetCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(internetCounts),
                    datasets: [{
                        data: Object.values(internetCounts),
                        backgroundColor: '#06b6d4',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });

        } catch (err) {
            console.error("Error loading data profile stats:", err);
        }
    }

    // -------------------------------------------------------------
    // 5. Real-Time "What-If" Sandbox Simulator
    // -------------------------------------------------------------
    const presets = {
        loyal: {
            gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes", tenure: 65,
            PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes",
            OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No",
            StreamingMovies: "Yes", Contract: "Two year", PaperlessBilling: "No",
            PaymentMethod: "Credit card (automatic)", MonthlyCharges: 68.80, TotalCharges: 4472.00
        },
        churn: {
            gender: "Female", SeniorCitizen: "1", Partner: "No", Dependents: "No", tenure: 3,
            PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No",
            OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
            StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
            PaymentMethod: "Electronic check", MonthlyCharges: 95.75, TotalCharges: 287.25
        }
    };

    // Load presets
    document.getElementById("btn-preset-loyal").addEventListener("click", () => populateSandboxForm(presets.loyal));
    document.getElementById("btn-preset-churn").addEventListener("click", () => populateSandboxForm(presets.churn));

    function populateSandboxForm(data) {
        const form = document.getElementById("predictionForm");
        Object.entries(data).forEach(([key, val]) => {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = val;
                // Update range values text
                if (key === "tenure" || key === "MonthlyCharges" || key === "TotalCharges") {
                    const spanId = key === "tenure" ? "tenure-val" : (key === "MonthlyCharges" ? "monthly-val" : "total-val");
                    document.getElementById(spanId).innerText = val;
                }
            }
        });
        triggerSandboxPrediction();
    }

    // Listen for inputs and sliders adjustments
    const sandboxForm = document.getElementById("predictionForm");
    
    // Sliders display update
    sandboxForm.querySelectorAll("input[type='range']").forEach(slider => {
        slider.addEventListener("input", () => {
            const name = slider.name;
            const val = slider.value;
            if (name === "tenure") {
                document.getElementById("tenure-val").innerText = val;
            } else if (name === "MonthlyCharges") {
                document.getElementById("monthly-val").innerText = parseFloat(val).toFixed(2);
            } else if (name === "TotalCharges") {
                document.getElementById("total-val").innerText = parseFloat(val).toFixed(2);
            }
        });
    });

    // Fire inference on input change events
    sandboxForm.querySelectorAll(".sandbox-input, .sandbox-input-slider").forEach(elem => {
        elem.addEventListener("change", triggerSandboxPrediction);
        elem.addEventListener("input", triggerSandboxPrediction);
    });

    // Store active record context for chatbot references
    let lastCustomerRecord = null;
    let lastXAIFactors = [];

    async function triggerSandboxPrediction() {
        const formData = new FormData(sandboxForm);
        const data = {};
        formData.forEach((value, key) => {
            if (key === "SeniorCitizen" || key === "tenure") {
                data[key] = parseInt(value, 10);
            } else if (key === "MonthlyCharges" || key === "TotalCharges") {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });

        lastCustomerRecord = data;

        try {
            // Run Primary single prediction + XAI
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            const resultBox = document.getElementById("result");
            const innerContainer = document.getElementById("result-inner-container");
            const errorContainer = document.getElementById("result-error-container");

            if (result.success) {
                innerContainer.style.display = "flex";
                errorContainer.style.display = "none";

                const label = result.prediction.churn_prediction;
                const prob = result.prediction.churn_probability;
                const pct = (prob * 100).toFixed(1);

                // Update text elements
                document.getElementById("res-label").innerText = label;
                document.getElementById("res-probability").innerText = pct + "%";
                
                // Update SVG Gauge circle stroke
                const circumference = 440; // 2 * pi * r (r=70)
                const offset = circumference - (prob * circumference);
                const gaugeFill = document.getElementById("gaugeFillCircle");
                gaugeFill.style.strokeDashoffset = offset;
                
                // Update badge and colors based on Risk Level
                const badge = document.getElementById("res-badge");
                if (label === "Yes") {
                    resultBox.style.backgroundColor = "rgba(239, 68, 68, 0.05)";
                    resultBox.style.borderColor = "rgba(239, 68, 68, 0.15)";
                    gaugeFill.style.stroke = "var(--danger)";
                    badge.className = "badge badge-danger";
                    badge.innerText = "HIGH RISK";
                } else {
                    resultBox.style.backgroundColor = "rgba(16, 185, 129, 0.05)";
                    resultBox.style.borderColor = "rgba(16, 185, 129, 0.15)";
                    gaugeFill.style.stroke = "var(--success)";
                    badge.className = "badge badge-success";
                    badge.innerText = "LOW RISK";
                }

                // Render XAI waterfall contributors
                if (result.explanations && result.explanations.length > 0) {
                    lastXAIFactors = result.explanations;
                    renderXAIChart(result.explanations);
                    renderAdvisorRecommendations(result.explanations, prob);
                }
            } else {
                innerContainer.style.display = "none";
                errorContainer.style.display = "block";
                document.getElementById("result-error-message").innerText = result.message || "An unexpected error occurred.";
            }

            // Fetch Multi-Model predictions comparisons
            const compRes = await fetch("/api/predict-all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const compData = await compRes.json();
            if (compData.success) {
                renderCompareChart(compData.predictions);
            }

        } catch (err) {
            console.error("Sandbox real-time prediction API error:", err);
        }
    }

    // Render XAI Waterfall Chart
    function renderXAIChart(explanations) {
        const ctx = document.getElementById("xaiChart");
        if (!ctx) return;

        const labels = explanations.map(e => e.feature);
        const values = explanations.map(e => e.contribution * 100);

        const colors = values.map(val => val > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)');
        const borderColors = values.map(val => val > 0 ? 'rgba(239, 68, 68, 1)' : 'rgba(16, 185, 129, 1)');

        if (xaiChartInstance) {
            xaiChartInstance.destroy();
        }

        xaiChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#94a3b8', font: { size: 9 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { size: 9 } }
                    }
                }
            }
        });
    }

    // Render Multi-Model Benchmarks Chart
    function renderCompareChart(predictions) {
        const ctx = document.getElementById("compareChart");
        if (!ctx) return;

        const labels = Object.keys(predictions);
        const values = Object.values(predictions).map(v => v * 100);

        if (compareChartInstance) {
            compareChartInstance.destroy();
        }

        compareChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: ['rgba(6, 182, 212, 0.7)', 'rgba(59, 130, 246, 0.7)', 'rgba(139, 92, 246, 0.7)'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 9 } } },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#94a3b8', font: { size: 9 } },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    // Render AI Advisor Interventions list
    function renderAdvisorRecommendations(explanations, baseProb) {
        const list = document.getElementById("advisor-recommendations-list");
        if (!list) return;

        list.innerHTML = "";

        // Sort positive risk contributions
        const criticalFactors = explanations
            .filter(e => e.contribution > 0.02)
            .sort((a, b) => b.contribution - a.contribution);

        if (criticalFactors.length === 0 || baseProb < 0.25) {
            list.innerHTML = `
                <li class="advisor-rec-item" style="color: var(--success); border: none;">
                    <span>No priority interventions needed. Customer retention profile is strong.</span>
                </li>
            `;
            return;
        }

        criticalFactors.forEach(factor => {
            const li = document.createElement("li");
            li.className = "advisor-rec-item";
            
            let action = "";
            let savings = (factor.contribution * 100).toFixed(0);

            if (factor.feature === "Contract") {
                action = "Upgrade to 1 or 2 Year Contract";
            } else if (factor.feature === "tenure") {
                action = "Apply loyalty tenure credits";
            } else if (factor.feature === "InternetService") {
                action = "Apply Fiber optic discount rate";
            } else if (factor.feature === "TechSupport") {
                action = "Offer complimentary Tech Support Bundle";
            } else if (factor.feature === "MonthlyCharges") {
                action = "Apply 15% billing discount credit";
            }

            li.innerHTML = `
                <span>${action}</span>
                <span style="font-weight: 700; color: var(--success); flex-shrink:0;">Risk Down ${savings}%</span>
            `;
            list.appendChild(li);
        });
    }

    // Load initial predictor values
    presets.loyal.MonthlyCharges = 68.80; // ensure floats
    populateSandboxForm(presets.loyal);

    // -------------------------------------------------------------
    // 6. Interactive AI Assistant Chatbot
    // -------------------------------------------------------------
    const chatInput = document.getElementById("chatInput");
    const chatSendBtn = document.getElementById("btnSendChat");
    const chatMessages = document.getElementById("chatMessages");

    chatSendBtn.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            sendChatMessage();
        }
    });

    // Shortcuts listeners
    document.getElementById("btn-chat-email").addEventListener("click", () => triggerChatShortcut("Draft a discount email for this customer"));
    document.getElementById("btn-chat-script").addEventListener("click", () => triggerChatShortcut("Give me an agent negotiation call script"));

    function triggerChatShortcut(promptText) {
        chatInput.value = promptText;
        sendChatMessage();
    }

    async function sendChatMessage() {
        const query = chatInput.value.trim();
        if (!query) return;

        // Clear input
        chatInput.value = "";

        // Append User Message to conversation
        appendMessage(query, "user");

        // Loading indicator
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "chat-message bot";
        loadingDiv.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Writing response...';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch("/api/chat-advisor", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: query,
                    customer_data: lastCustomerRecord || {},
                    xai_factors: lastXAIFactors || []
                })
            });
            const data = await res.json();
            
            // Remove loading
            chatMessages.removeChild(loadingDiv);

            if (data.success) {
                // Parse markdown to HTML
                const parsedText = parseSimpleMarkdown(data.reply);
                appendMessage(parsedText, "bot", true);
            } else {
                appendMessage("Failed to get chatbot response.", "bot");
            }
        } catch (err) {
            chatMessages.removeChild(loadingDiv);
            console.error("Chatbot advisor fetch error:", err);
            appendMessage("Connection error with AI service.", "bot");
        }
    }

    function appendMessage(text, sender, isHTML = false) {
        const div = document.createElement("div");
        div.className = `chat-message ${sender}`;
        if (isHTML) {
            div.innerHTML = text;
        } else {
            div.innerText = text;
        }
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Simple markdown-to-html compiler for simulated chat responses
    function parseSimpleMarkdown(text) {
        let lines = text.split("\n");
        let html = "";
        let inList = false;

        lines.forEach(line => {
            if (line.startsWith("### ")) {
                if (inList) { html += "</ul>"; inList = false; }
                html += `<h4 style="margin: 10px 0 6px 0; color: white; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 4px;">${line.replace("### ", "")}</h4>`;
            } else if (line.startsWith("#### ")) {
                if (inList) { html += "</ul>"; inList = false; }
                html += `<h5 style="margin: 8px 0 4px 0; color: var(--accent); font-weight: 700;">${line.replace("#### ", "")}</h5>`;
            } else if (line.startsWith("* ") || line.startsWith("- ")) {
                if (!inList) { html += '<ul style="margin-left: 16px; margin-bottom: 8px;">'; inList = true; }
                // Parse bolding inside items
                let parsedLine = line.replace(/^\* |^- /, "");
                parsedLine = parsedLine.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                html += `<li>${parsedLine}</li>`;
            } else if (line.startsWith("> ")) {
                if (inList) { html += "</ul>"; inList = false; }
                let quoteText = line.replace("> ", "");
                quoteText = quoteText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                html += `<blockquote style="border-left: 2px solid var(--accent); padding-left: 10px; color: var(--text-secondary); margin: 8px 0; font-style: italic;">${quoteText}</blockquote>`;
            } else {
                if (inList) { html += "</ul>"; inList = false; }
                if (line.trim().length > 0) {
                    let parsedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    html += `<p style="margin-bottom: 6px;">${parsedLine}</p>`;
                }
            }
        });
        if (inList) html += "</ul>";
        return html;
    }

    // -------------------------------------------------------------
    // 7. Batch CSV Upload Inference & Graphing
    // -------------------------------------------------------------
    const uploadZone = document.getElementById("uploadZone");
    const csvFileInput = document.getElementById("csvFileInput");
    let batchPredictionsData = null; // Stored predicted array for export download

    uploadZone.addEventListener("click", () => csvFileInput.click());
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleBatchUpload(files[0]);
        }
    });

    csvFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleBatchUpload(e.target.files[0]);
        }
    });

    async function handleBatchUpload(file) {
        if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
            alert("Please upload a valid CSV formatted file.");
            return;
        }

        // Show/Hide blocks
        uploadZone.style.display = "none";
        document.getElementById("batch-spinner").style.display = "block";
        document.getElementById("batch-results-wrapper").style.display = "none";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/predict-batch", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            document.getElementById("batch-spinner").style.display = "none";

            if (data.success) {
                batchPredictionsData = data.predictions;
                document.getElementById("batch-results-wrapper").style.display = "block";

                // Populate KPIs
                document.getElementById("batch-total").innerText = data.summary.total_customers;
                document.getElementById("batch-churn-count").innerText = data.summary.churned_count;
                document.getElementById("batch-retain-count").innerText = data.summary.retained_count;
                document.getElementById("batch-churn-rate").innerText = data.summary.churn_rate + "%";

                // Populate table preview rows (limit to first 10 rows for display)
                const tbody = document.getElementById("batch-prediction-rows");
                tbody.innerHTML = "";
                const previewRecords = data.predictions.slice(0, 10);
                previewRecords.forEach(pred => {
                    const row = document.createElement("tr");
                    const isChurn = pred.churn_prediction === "Yes";
                    row.innerHTML = `
                        <td><code>${pred.customerID}</code></td>
                        <td><span class="badge ${isChurn ? 'badge-danger' : 'badge-success'}">${pred.churn_prediction}</span></td>
                        <td style="font-weight: 700; color: white;">${(pred.churn_probability * 100).toFixed(1)}%</td>
                    `;
                    tbody.appendChild(row);
                });

                // Cohort Split Donut Chart drawing
                if (batchChartInstance) {
                    batchChartInstance.destroy();
                }
                const ctx = document.getElementById("batchChart").getContext("2d");
                batchChartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Retained', 'Churn Risk'],
                        datasets: [{
                            data: [data.summary.retained_count, data.summary.churned_count],
                            backgroundColor: ['#10b981', '#ef4444'],
                            borderColor: '#060913',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: '#94a3b8', font: { size: 12 } }
                            }
                        }
                    }
                });

            } else {
                uploadZone.style.display = "block";
                alert("Batch inference failed: " + data.message);
            }
        } catch (err) {
            document.getElementById("batch-spinner").style.display = "none";
            uploadZone.style.display = "block";
            console.error("Batch inference call failed:", err);
        }
    }

    // Export scored dataset as CSV file
    document.getElementById("btnDownloadBatch").addEventListener("click", () => {
        if (!batchPredictionsData) return;

        let csvContent = "customerID,churn_prediction,churn_probability\n";
        batchPredictionsData.forEach(row => {
            csvContent += `${row.customerID},${row.churn_prediction},${row.churn_probability}\n`;
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", "batch_churn_predictions.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // -------------------------------------------------------------
    // 8. Pipeline Trigger Actions & Logs Live Terminal
    // -------------------------------------------------------------
    const btnTriggerTrain = document.getElementById("btnTriggerTrain");
    const terminalLogs = document.getElementById("terminalLogs");
    let logIntervalId = null;

    btnTriggerTrain.addEventListener("click", async () => {
        btnTriggerTrain.disabled = true;
        btnTriggerTrain.innerHTML = '<i class="fa-solid fa-arrows-spin fa-spin"></i> Execution Running...';
        terminalLogs.innerText = "Initializing pipeline sequence...\nStarting Data Ingestion stage...";

        try {
            const res = await fetch("/api/trigger-train", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                // Start polling logs console
                startLogsPolling();
            } else {
                btnTriggerTrain.disabled = false;
                btnTriggerTrain.innerHTML = '<i class="fa-solid fa-play"></i> Run Training Pipeline';
                alert("Pipeline execution trigger failed: " + data.message);
            }
        } catch (err) {
            btnTriggerTrain.disabled = false;
            btnTriggerTrain.innerHTML = '<i class="fa-solid fa-play"></i> Run Training Pipeline';
            console.error("Training trigger failed:", err);
        }
    });

    function startLogsPolling() {
        if (logIntervalId) clearInterval(logIntervalId);
        
        logIntervalId = setInterval(async () => {
            try {
                const res = await fetch("/api/logs");
                const data = await res.json();
                if (data.success) {
                    terminalLogs.innerText = data.logs;
                    terminalLogs.scrollTop = terminalLogs.scrollHeight;

                    // Stop polling if pipeline completes inside logs text
                    if (data.logs.includes("TRAINING PIPELINE COMPLETE")) {
                        clearInterval(logIntervalId);
                        logIntervalId = null;
                        btnTriggerTrain.disabled = false;
                        btnTriggerTrain.innerHTML = '<i class="fa-solid fa-play"></i> Run Training Pipeline';
                        
                        // Hot reload dashboard metrics & importances
                        await loadMetrics();
                        loadFeaturesChart();
                    }
                }
            } catch (err) {
                console.error("Logs fetch failed:", err);
            }
        }, 1000);
    }

    // -------------------------------------------------------------
    // 9. Evidently AI Drift Monitoring
    // -------------------------------------------------------------
    const btnTriggerDrift = document.getElementById("btnTriggerDrift");
    const frameContainer = document.getElementById("monitoringFrameContainer");
    const iframe = document.getElementById("monitoringIframe");

    btnTriggerDrift.addEventListener("click", async () => {
        btnTriggerDrift.disabled = true;
        btnTriggerDrift.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating Drift...';

        try {
            const res = await fetch("/api/trigger-drift", { method: "POST" });
            const data = await res.json();
            
            btnTriggerDrift.disabled = false;
            btnTriggerDrift.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Run Drift Diagnostics';

            if (data.success) {
                frameContainer.style.display = "block";
                iframe.src = data.report_url;
            } else {
                alert("Drift analysis check failed: " + data.message);
            }
        } catch (err) {
            btnTriggerDrift.disabled = false;
            btnTriggerDrift.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Run Drift Diagnostics';
            console.error("Drift check failed:", err);
        }
    });

});
