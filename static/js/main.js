/**
 * AI-Powered Personal Finance Advisor - Client Application Logic
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Live AI Category Suggestion on Add Transaction
    const descInput = document.getElementById('txn-description');
    const catSelect = document.getElementById('txn-category');
    const suggestionBox = document.getElementById('ai-suggestion-box');
    const suggestionText = document.getElementById('ai-suggestion-text');
    const applySuggestionBtn = document.getElementById('apply-suggestion-btn');
    
    let debounceTimer = null;

    if (descInput && catSelect && suggestionBox) {
        descInput.addEventListener('input', function () {
            const query = descInput.value.trim();
            clearTimeout(debounceTimer);
            
            if (query.length < 3) {
                suggestionBox.style.display = 'none';
                return;
            }

            debounceTimer = setTimeout(() => {
                fetch(`/api/predict-category?desc=${encodeURIComponent(query)}`)
                    .then(res => res.json())
                    .then(data => {
                        if (data.category && data.confidence > 20) {
                            suggestionBox.style.display = 'flex';
                            suggestionText.innerHTML = `AI Suggests: <strong>${data.category}</strong> <span class="badge bg-secondary ms-1">${data.confidence}% confidence</span>`;
                            
                            // Auto-select if currently on "Auto"
                            if (catSelect.value === 'Auto' || !catSelect.value) {
                                catSelect.value = data.category;
                            }
                            
                            if (applySuggestionBtn) {
                                applySuggestionBtn.onclick = function (e) {
                                    e.preventDefault();
                                    catSelect.value = data.category;
                                    catSelect.classList.add('border-primary');
                                    setTimeout(() => catSelect.classList.remove('border-primary'), 1000);
                                };
                            }
                        } else {
                            suggestionBox.style.display = 'none';
                        }
                    })
                    .catch(err => console.error('Categorization error:', err));
            }, 300);
        });
    }

    // 2. Real-time Anomaly Warning on Add Transaction
    const amountInput = document.getElementById('txn-amount');
    const anomalyAlert = document.getElementById('ai-anomaly-alert');
    const anomalyMessage = document.getElementById('ai-anomaly-message');
    const typeRadios = document.querySelectorAll('input[name="type"]');

    function checkLiveAnomaly() {
        if (!amountInput || !catSelect || !anomalyAlert) return;
        
        let isExpense = true;
        typeRadios.forEach(r => {
            if (r.checked && r.value === 'income') isExpense = false;
        });

        if (!isExpense) {
            anomalyAlert.style.display = 'none';
            return;
        }

        const amt = parseFloat(amountInput.value);
        const cat = catSelect.value;

        if (amt > 0 && cat && cat !== 'Auto') {
            fetch('/api/check-anomaly', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount: amt, category: cat })
            })
            .then(res => res.json())
            .then(data => {
                if (data.is_anomaly) {
                    anomalyAlert.style.display = 'block';
                    anomalyMessage.textContent = data.warning;
                } else {
                    anomalyAlert.style.display = 'none';
                }
            })
            .catch(err => console.error('Anomaly check error:', err));
        } else {
            anomalyAlert.style.display = 'none';
        }
    }

    if (amountInput) {
        amountInput.addEventListener('blur', checkLiveAnomaly);
        if (catSelect) catSelect.addEventListener('change', checkLiveAnomaly);
    }

    // 3. Retrain Models Handler (ML Models Page)
    const retrainBtn = document.getElementById('retrain-models-btn');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', function () {
            const originalHtml = retrainBtn.innerHTML;
            retrainBtn.disabled = true;
            retrainBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Retraining Models...`;
            
            fetch('/api/retrain-models', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    alert('ML Models successfully retrained and validated!');
                    window.location.reload();
                })
                .catch(err => {
                    alert('Retraining failed: ' + err);
                    retrainBtn.disabled = false;
                    retrainBtn.innerHTML = originalHtml;
                });
        });
    }

    // 4. Interactive Live Classifier Sandbox
    const testDescInput = document.getElementById('sandbox-desc');
    const testBtn = document.getElementById('sandbox-predict-btn');
    const testResult = document.getElementById('sandbox-result');

    if (testBtn && testDescInput && testResult) {
        testBtn.addEventListener('click', function () {
            const desc = testDescInput.value.trim();
            if (!desc) return;
            
            fetch(`/api/predict-category?desc=${encodeURIComponent(desc)}`)
                .then(res => res.json())
                .then(data => {
                    testResult.style.display = 'block';
                    let html = `
                        <div class="alert alert-success d-flex justify-content-between align-items-center mb-3">
                            <div>
                                <h6 class="mb-0 fw-bold">Predicted Category: <span class="text-primary">${data.category}</span></h6>
                                <small class="text-muted">Prediction Confidence: ${data.confidence}%</small>
                            </div>
                            <span class="badge bg-primary px-3 py-2 fs-6">${data.category}</span>
                        </div>
                        <h6 class="text-muted mb-2 font-monospace fs-7">Class Probability Distribution:</h6>
                        <div class="row g-2">
                    `;
                    for (const [c, p] of Object.entries(data.probabilities || {})) {
                        const pct = (p * 100).toFixed(1);
                        html += `
                            <div class="col-md-3 col-6">
                                <div class="p-2 border rounded bg-white">
                                    <div class="d-flex justify-content-between small mb-1">
                                        <span>${c}</span>
                                        <span class="fw-bold">${pct}%</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar ${c === data.category ? 'bg-primary' : 'bg-secondary'}" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    html += `</div>`;
                    testResult.innerHTML = html;
                });
        });
    }
});

// Chart.js helper functions
window.renderOverviewChart = function (canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    fetch('/api/chart-data/overview')
        .then(res => res.json())
        .then(data => {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [
                        {
                            label: 'Income (₹)',
                            data: data.income,
                            backgroundColor: 'rgba(16, 185, 129, 0.8)',
                            borderRadius: 6,
                            barPercentage: 0.7
                        },
                        {
                            label: 'Expenses (₹)',
                            data: data.expense,
                            backgroundColor: 'rgba(239, 68, 68, 0.8)',
                            borderRadius: 6,
                            barPercentage: 0.7
                        },
                        {
                            type: 'line',
                            label: 'Net Savings (₹)',
                            data: data.savings,
                            borderColor: '#4f46e5',
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            borderWidth: 3,
                            tension: 0.35,
                            fill: false,
                            pointRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    return `${context.dataset.label}: ₹${context.parsed.y.toLocaleString('en-IN')}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#f1f5f9' },
                            ticks: {
                                callback: val => '₹' + (val / 1000).toFixed(0) + 'k'
                            }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });
        });
};

window.renderCategoryChart = function (canvasId, monthFilter) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const url = monthFilter ? `/api/chart-data/category?month=${monthFilter}` : '/api/chart-data/category';
    fetch(url)
        .then(res => res.json())
        .then(data => {
            const colors = [
                '#ef4444', '#f59e0b', '#6366f1', '#64748b',
                '#3b82f6', '#a855f7', '#10b981', '#94a3b8'
            ];
            
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: colors.slice(0, data.labels.length),
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { boxWidth: 12, padding: 12 } },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const pct = ((context.parsed / total) * 100).toFixed(1);
                                    return `${context.label}: ₹${context.parsed.toLocaleString('en-IN')} (${pct}%)`;
                                }
                            }
                        }
                    },
                    cutout: '68%'
                }
            });
        });
};
