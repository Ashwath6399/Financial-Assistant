// Trading Journal Dashboard - Interactive JavaScript

// =============================================================================
// GLOBAL STATE
// =============================================================================
let pnlChart = null;
let mlChart = null;
let accountPnlChart = null;
let drawdownChart = null;
let tradePnlDistChart = null;
let currentAccountData = null;
let overviewData = null;

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    loadOverviewData();
    loadAccounts();
});

// =============================================================================
// TAB NAVIGATION
// =============================================================================
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            if (this.disabled) return;

            const tabId = this.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-tab') === tabId) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Load data for specific tabs if needed
    if (tabId === 'overview' && !overviewData) {
        loadOverviewData();
    }

    if (tabId === 'ml-models' && !mlData) {
        loadMLData();
    }

    if (tabId === 'trades' && !tradesData) {
        loadTradesData();
    }

    if (tabId === 'screener' && !screenerData) {
        loadScreenerData();
    }

    if (tabId === 'market' && !marketData) {
        loadMarketData();
    }

    if (tabId === 'database' && databaseTables.length === 0) {
        loadDatabaseData();
    }
}

// =============================================================================
// OVERVIEW TAB
// =============================================================================
async function loadOverviewData() {
    try {
        const response = await fetch('/api/overview/summary');
        const data = await response.json();
        overviewData = data;

        updateOverviewStats(data);
        updateHighlights(data);
        updateMLSummary(data);
        updateBenchmarkSummary(data);
        updateDataCoverage(data);
        createOverviewCharts(data);

    } catch (error) {
        console.error('Error loading overview data:', error);
    }
}

function updateOverviewStats(data) {
    const portfolio = data.portfolio;

    // Total P&L
    const pnlElement = document.getElementById('overview-total-pnl');
    pnlElement.textContent = formatCurrency(portfolio.total_pnl);
    pnlElement.className = portfolio.total_pnl >= 0 ? 'stat-value value-positive' : 'stat-value value-negative';

    // Total Trades
    document.getElementById('overview-total-trades').textContent =
        portfolio.total_trades ? portfolio.total_trades.toLocaleString() : '0';

    // Total Accounts
    document.getElementById('overview-total-accounts').textContent = portfolio.total_accounts || 0;

    // Avg Win Rate (stored as decimal 0.58, display as 58%)
    const avgWinRate = portfolio.avg_win_rate * 100;
    const winRateElement = document.getElementById('overview-avg-winrate');
    winRateElement.textContent = formatPercent(avgWinRate);
    winRateElement.className = avgWinRate >= 50 ? 'stat-value value-positive' : 'stat-value value-negative';
}

function updateHighlights(data) {
    // Best Account (win_rate stored as decimal, multiply by 100)
    document.getElementById('best-account-id').textContent = data.best_account.account_id;
    document.getElementById('best-account-pnl').textContent = formatCurrency(data.best_account.net_pnl);
    document.getElementById('best-account-winrate').textContent = `Win Rate: ${formatPercent(data.best_account.win_rate * 100)}`;

    // Worst Account (win_rate stored as decimal, multiply by 100)
    document.getElementById('worst-account-id').textContent = data.worst_account.account_id;
    document.getElementById('worst-account-pnl').textContent = formatCurrency(data.worst_account.net_pnl);
    document.getElementById('worst-account-winrate').textContent = `Win Rate: ${formatPercent(data.worst_account.win_rate * 100)}`;

    // Biggest Win
    document.getElementById('biggest-win-symbol').textContent = data.biggest_win.symbol;
    document.getElementById('biggest-win-amount').textContent = formatCurrency(data.biggest_win.amount);
    document.getElementById('biggest-win-account').textContent = `Account: ${data.biggest_win.account_id}`;

    // Biggest Loss
    document.getElementById('biggest-loss-symbol').textContent = data.biggest_loss.symbol;
    document.getElementById('biggest-loss-amount').textContent = formatCurrency(data.biggest_loss.amount);
    document.getElementById('biggest-loss-account').textContent = `Account: ${data.biggest_loss.account_id}`;
}

function updateMLSummary(data) {
    const ml = data.ml_stats;

    document.getElementById('ml-predictions-count').textContent =
        ml.total_predictions ? ml.total_predictions.toLocaleString() : '0';
    document.getElementById('ml-overall-accuracy').textContent = formatPercent(ml.overall_accuracy);
    document.getElementById('ml-take-accuracy').textContent = formatPercent(ml.take_accuracy);
    document.getElementById('ml-avoid-accuracy').textContent = formatPercent(ml.avoid_accuracy);
}

function updateBenchmarkSummary(data) {
    const benchmark = data.benchmark;

    document.getElementById('benchmark-beat-count').textContent =
        `${benchmark.accounts_beat_market} / ${benchmark.total_accounts}`;
    document.getElementById('avg-sharpe').textContent = formatNumber(data.portfolio.avg_sharpe, 2);
    document.getElementById('avg-outperformance').textContent = formatPercent(benchmark.avg_outperformance);
    document.getElementById('price-data-count').textContent =
        data.data_coverage.price_count ? data.data_coverage.price_count.toLocaleString() : '0';
}

function updateDataCoverage(data) {
    const coverage = data.data_coverage;

    document.getElementById('data-symbols').textContent = coverage.symbols || 0;
    document.getElementById('data-date-range').textContent = coverage.date_range || '--';
    document.getElementById('data-price-points').textContent =
        coverage.price_count ? coverage.price_count.toLocaleString() : '0';
    document.getElementById('data-fills').textContent =
        coverage.fills_count ? coverage.fills_count.toLocaleString() : '0';
}

function createOverviewCharts(data) {
    createAccountPnlChart(data.all_accounts);
}

function createAccountPnlChart(accounts) {
    const ctx = document.getElementById('accountPnlChart');
    if (!ctx) return;

    if (accountPnlChart) {
        accountPnlChart.destroy();
    }

    // Sort by P&L and take top 15
    const sortedAccounts = [...accounts].sort((a, b) => b.net_pnl - a.net_pnl).slice(0, 15);

    const labels = sortedAccounts.map(a => a.account_id);
    const pnlData = sortedAccounts.map(a => a.net_pnl);
    const colors = pnlData.map(p => p >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)');
    const borderColors = pnlData.map(p => p >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)');

    accountPnlChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Net P&L',
                data: pnlData,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'P&L: ' + formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45 },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                }
            }
        }
    });
}

// =============================================================================
// ACCOUNT PERFORMANCE TAB
// =============================================================================
async function loadAccounts() {
    try {
        const response = await fetch('/api/accounts');
        const accounts = await response.json();

        const select = document.getElementById('accountSelect');
        select.innerHTML = '<option value="">Select an account...</option>';

        accounts.forEach(account => {
            const option = document.createElement('option');
            option.value = account.account_id;
            option.textContent = account.account_id;
            select.appendChild(option);
        });

        select.addEventListener('change', (e) => {
            if (e.target.value) {
                loadAccountData(e.target.value);
            } else {
                document.getElementById('accountDashboard').style.display = 'none';
                document.getElementById('noAccountSelected').style.display = 'flex';
            }
        });
    } catch (error) {
        console.error('Error loading accounts:', error);
    }
}

async function loadAccountData(accountId) {
    document.getElementById('noAccountSelected').style.display = 'none';
    document.getElementById('accountDashboard').style.display = 'block';

    try {
        const response = await fetch(`/api/account/${accountId}`);
        const data = await response.json();
        currentAccountData = data;

        updateMetrics(data.metrics, data.risk);
        updateBenchmark(data.benchmark);
        updateCharts(data);

    } catch (error) {
        console.error('Error loading account data:', error);
    }
}

function updateMetrics(metrics, risk) {
    if (!metrics) return;

    // Primary Metrics
    // Net P&L
    updateMetricCard('net_pnl', formatCurrency(metrics.net_pnl), metrics.net_pnl >= 0);

    // Win Rate (stored as decimal, convert to percentage)
    const winRateValue = metrics.win_rate * 100;
    updateMetricCard('win_rate', formatPercent(winRateValue), winRateValue >= 50);

    // Total Trades
    updateMetricCard('total_trades', metrics.total_trades, true);

    // Calculate derived metrics
    const totalTrades = metrics.total_trades || 0;
    const winRate = metrics.win_rate || 0;
    const winningTrades = Math.round(totalTrades * winRate);
    const losingTrades = totalTrades - winningTrades;
    const avgWin = metrics.avg_win || 0;
    const avgLoss = Math.abs(metrics.avg_loss || 0);
    const grossProfit = winningTrades * avgWin;
    const grossLoss = losingTrades * avgLoss;
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;
    const expectancy = (winRate * avgWin) - ((1 - winRate) * avgLoss);

    // Profit Factor
    updateMetricCard('profit_factor', formatNumber(profitFactor, 2), profitFactor >= 1);

    // Risk-Adjusted Returns
    // Sharpe Ratio
    updateMetricCard('sharpe_ratio', formatNumber(metrics.sharpe_ratio, 2), metrics.sharpe_ratio >= 1);

    // Sortino Ratio
    updateMetricCard('sortino_ratio', formatNumber(metrics.sortino_ratio, 2), metrics.sortino_ratio >= 1);

    // Calmar Ratio
    updateMetricCard('calmar_ratio', formatNumber(metrics.calmar_ratio, 2), metrics.calmar_ratio >= 1);

    // Risk/Reward Ratio
    updateMetricCard('risk_reward_ratio', formatNumber(metrics.risk_reward_ratio, 2), metrics.risk_reward_ratio >= 1);

    // Risk Metrics
    // Max Drawdown
    updateMetricCard('max_drawdown', formatPercent(metrics.max_drawdown), false);

    // VaR 95%
    if (risk && risk.var_95_pct !== undefined) {
        updateMetricCard('var_95_pct', formatPercent(risk.var_95_pct), false);
    }

    // Volatility - stored in benchmark as decimal, need to multiply by 100
    // Will be updated by updateBenchmark function

    // Largest Loss
    updateMetricCard('largest_loss', formatCurrency(metrics.largest_loss), false);

    // Trade Statistics - calculated values
    document.getElementById('metric-winning_trades').textContent = winningTrades;
    document.getElementById('metric-losing_trades').textContent = losingTrades;
    document.getElementById('metric-avg_win').textContent = formatCurrency(avgWin);
    document.getElementById('metric-avg_loss').textContent = formatCurrency(-avgLoss);
    document.getElementById('metric-largest_win').textContent = formatCurrency(metrics.largest_win);
    document.getElementById('metric-gross_profit').textContent = formatCurrency(grossProfit);
    document.getElementById('metric-gross_loss').textContent = formatCurrency(-grossLoss);

    // Expectancy
    const expectancyElement = document.getElementById('metric-expectancy');
    expectancyElement.textContent = formatCurrency(expectancy);
    expectancyElement.className = expectancy >= 0 ? 'metric-value positive' : 'metric-value negative';
}

function updateMetricCard(metricName, value, isPositive) {
    const valueElement = document.getElementById(`metric-${metricName}`);
    const changeElement = document.getElementById(`change-${metricName}`);

    if (valueElement) {
        valueElement.textContent = value;

        if (metricName === 'net_pnl' || metricName === 'win_rate' || metricName === 'sharpe_ratio' ||
            metricName === 'sortino_ratio' || metricName === 'risk_reward_ratio') {
            valueElement.style.color = isPositive ? 'var(--success-color)' : 'var(--danger-color)';
        } else if (metricName === 'max_drawdown' || metricName === 'var_95_pct') {
            valueElement.style.color = 'var(--danger-color)';
        }
    }

    if (changeElement && metricName !== 'total_trades') {
        changeElement.textContent = isPositive ? '^ Good' : 'v Poor';
        changeElement.className = `metric-change ${isPositive ? 'positive' : 'negative'}`;
    }
}

function updateBenchmark(benchmark) {
    if (!benchmark) return;

    document.getElementById('benchmark-account_return').textContent =
        formatPercent(benchmark.account_return_pct);
    document.getElementById('benchmark-spy_return').textContent =
        formatPercent(benchmark.benchmark_return_pct);

    const outperformance = benchmark.outperformance_pct;
    const outElement = document.getElementById('benchmark-outperformance');
    outElement.textContent = formatPercent(outperformance);
    outElement.style.color = outperformance >= 0 ? 'var(--success-color)' : 'var(--danger-color)';

    const beatElement = document.getElementById('benchmark-beat_market');
    beatElement.textContent = benchmark.beat_market ? 'Yes' : 'No';
    beatElement.style.color = benchmark.beat_market ? 'var(--success-color)' : 'var(--danger-color)';

    // Additional benchmark metrics
    const accountSharpeElement = document.getElementById('benchmark-account_sharpe');
    if (accountSharpeElement && benchmark.account_sharpe_ratio !== undefined) {
        accountSharpeElement.textContent = formatNumber(benchmark.account_sharpe_ratio, 2);
    }

    const spySharpeElement = document.getElementById('benchmark-spy_sharpe');
    if (spySharpeElement && benchmark.benchmark_sharpe_ratio !== undefined) {
        spySharpeElement.textContent = formatNumber(benchmark.benchmark_sharpe_ratio, 2);
    }

    // Update volatility from benchmark (stored as decimal, display as %)
    const volatilityElement = document.getElementById('metric-volatility');
    if (volatilityElement && benchmark.account_volatility !== undefined) {
        volatilityElement.textContent = formatPercent(benchmark.account_volatility * 100);
        volatilityElement.style.color = 'var(--danger-color)';
    }
}

function updateCharts(data) {
    updatePnLChart(data.timeline);
    updateDrawdownChart(data.drawdown);
    updateMLChart(data.ml_summary);
    updateTradePnLDistChart(data.pnl_distribution);
    updateMonthlyReturnsHeatmap(data.monthly_returns);
    updateStreaks(data.risk);
}

function updatePnLChart(timeline) {
    const ctx = document.getElementById('pnlChart');
    if (!ctx) return;

    if (pnlChart) {
        pnlChart.destroy();
    }

    const labels = timeline.map(t => new Date(t.date).toLocaleDateString());
    const data = timeline.map(t => t.cumulative_pnl);

    pnlChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cumulative P&L',
                data: data,
                borderColor: 'rgba(99, 102, 241, 1)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'P&L: ' + formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                }
            }
        }
    });
}

function updateMLChart(mlSummary) {
    const ctx = document.getElementById('mlRecommendationsChart');
    if (!ctx) return;

    if (mlChart) {
        mlChart.destroy();
    }

    if (!mlSummary || mlSummary.length === 0) return;

    const labels = mlSummary.map(s => s.recommendation);
    const pnls = mlSummary.map(s => s.total_pnl);
    const accuracies = mlSummary.map(s => s.accuracy);

    mlChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Total P&L ($)',
                    data: pnls,
                    backgroundColor: pnls.map(p => p >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'),
                    borderColor: pnls.map(p => p >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'),
                    borderWidth: 2,
                    yAxisID: 'y',
                    borderRadius: 4
                },
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    type: 'line',
                    borderColor: 'rgba(245, 158, 11, 1)',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    yAxisID: 'y1',
                    pointRadius: 6,
                    pointBackgroundColor: 'rgba(245, 158, 11, 1)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.label.includes('P&L')) {
                                return 'P&L: ' + formatCurrency(context.parsed.y);
                            } else {
                                return 'Accuracy: ' + context.parsed.y.toFixed(2) + '%';
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value.toFixed(0) + '%';
                        }
                    },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function updateDrawdownChart(drawdownData) {
    const ctx = document.getElementById('drawdownChart');
    if (!ctx) return;

    if (drawdownChart) {
        drawdownChart.destroy();
    }

    if (!drawdownData || drawdownData.length === 0) return;

    const labels = drawdownData.map(d => new Date(d.date).toLocaleDateString());
    const data = drawdownData.map(d => d.drawdown);

    drawdownChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Drawdown %',
                data: data,
                borderColor: 'rgba(239, 68, 68, 1)',
                backgroundColor: 'rgba(239, 68, 68, 0.3)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Drawdown: ' + context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' },
                    max: 0
                }
            }
        }
    });
}

function updateTradePnLDistChart(distribution) {
    const ctx = document.getElementById('tradePnlDistChart');
    if (!ctx) return;

    if (tradePnlDistChart) {
        tradePnlDistChart.destroy();
    }

    if (!distribution || distribution.length === 0) return;

    const labels = distribution.map(d => '$' + d.range.toLocaleString());
    const data = distribution.map(d => d.count);
    const colors = distribution.map(d => d.range >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');

    tradePnlDistChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of Trades',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + ' trades';
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45, maxTicksLimit: 15 },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                }
            }
        }
    });
}

function updateMonthlyReturnsHeatmap(monthlyReturns) {
    const container = document.getElementById('monthlyReturnsHeatmap');
    if (!container) return;

    if (!monthlyReturns || Object.keys(monthlyReturns).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No monthly data available</p>';
        return;
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const years = Object.keys(monthlyReturns).sort();

    let html = '<table class="heatmap-table"><thead><tr><th>Year</th>';
    months.forEach(m => html += `<th>${m}</th>`);
    html += '<th>Total</th></tr></thead><tbody>';

    years.forEach(year => {
        html += `<tr><td><strong>${year}</strong></td>`;
        let yearTotal = 0;
        for (let m = 1; m <= 12; m++) {
            const value = monthlyReturns[year][m];
            if (value !== undefined) {
                yearTotal += value;
                const colorClass = getHeatmapColorClass(value);
                html += `<td class="heatmap-cell ${colorClass}">${formatCurrency(value)}</td>`;
            } else {
                html += '<td class="heatmap-cell no-data">--</td>';
            }
        }
        const totalColorClass = getHeatmapColorClass(yearTotal);
        html += `<td class="heatmap-cell ${totalColorClass}"><strong>${formatCurrency(yearTotal)}</strong></td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function getHeatmapColorClass(value) {
    if (value < -500) return 'negative-3';
    if (value < -200) return 'negative-2';
    if (value < 0) return 'negative-1';
    if (value === 0) return 'neutral';
    if (value < 200) return 'positive-1';
    if (value < 500) return 'positive-2';
    return 'positive-3';
}

function updateStreaks(risk) {
    if (!risk) return;

    // Current streak - positive means win streak, negative means loss streak
    const currentStreak = risk.current_streak || 0;
    if (currentStreak >= 0) {
        document.getElementById('current-win-streak').textContent = currentStreak;
        document.getElementById('current-loss-streak').textContent = 0;
    } else {
        document.getElementById('current-win-streak').textContent = 0;
        document.getElementById('current-loss-streak').textContent = Math.abs(currentStreak);
    }

    document.getElementById('max-win-streak').textContent = risk.max_win_streak || 0;
    document.getElementById('max-loss-streak').textContent = risk.max_loss_streak || 0;
}

// =============================================================================
// MODAL
// =============================================================================
async function showMetricFormula(metricName) {
    try {
        const response = await fetch(`/api/metric/formula/${metricName}`);
        const formula = await response.json();

        document.getElementById('formula-name').textContent = formula.name;
        document.getElementById('formula-formula').textContent = formula.formula;
        document.getElementById('formula-explanation').textContent = formula.explanation;
        document.getElementById('formula-interpretation').textContent = formula.interpretation;

        document.getElementById('formulaModal').style.display = 'block';
    } catch (error) {
        console.error('Error loading formula:', error);
    }
}

function closeModal() {
    document.getElementById('formulaModal').style.display = 'none';
}

window.onclick = function(event) {
    const modal = document.getElementById('formulaModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

// =============================================================================
// FORMATTING UTILITIES
// =============================================================================
function formatCurrency(value) {
    if (value === null || value === undefined) return '$0.00';
    const formatted = Math.abs(value).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    return value < 0 ? '-$' + formatted : '$' + formatted;
}

function formatPercent(value) {
    if (value === null || value === undefined) return '0.00%';
    return value.toFixed(2) + '%';
}

function formatNumber(value, decimals = 0) {
    if (value === null || value === undefined) return '0';
    return value.toFixed(decimals);
}

// =============================================================================
// TAB 3: ML MODELS
// =============================================================================
let mlData = null;
let mlMonthlyAccuracyChart = null;
let mlSymbolChart = null;
let allPredictionsData = [];
let predictionsCurrentPage = 1;
const PREDICTIONS_PER_PAGE = 20;

async function loadMLData() {
    try {
        const response = await fetch('/api/ml/summary');
        const data = await response.json();
        mlData = data;

        updateModelCards(data);
        updateConfusionMatrices(data);
        updateRecommendations(data);
        updateConfidenceLevels(data);
        createMLCharts(data);

        // Store all predictions and initialize pagination
        allPredictionsData = data.recent_predictions || [];
        predictionsCurrentPage = 1;
        renderPredictionsPage();

    } catch (error) {
        console.error('Error loading ML data:', error);
    }
}

function updateModelCards(data) {
    const metrics = data.model_metrics;
    const overall = data.overall;

    // XGBoost metrics
    if (metrics.xgboost) {
        document.getElementById('xgboost-accuracy').textContent = formatPercent(metrics.xgboost.accuracy);
        document.getElementById('xgboost-precision').textContent = formatPercent(metrics.xgboost.precision);
        document.getElementById('xgboost-recall').textContent = formatPercent(metrics.xgboost.recall);
        document.getElementById('xgboost-f1').textContent = formatPercent(metrics.xgboost.f1_score);
    }

    // Hybrid metrics
    if (metrics.hybrid) {
        document.getElementById('hybrid-accuracy').textContent = formatPercent(metrics.hybrid.accuracy);
        document.getElementById('hybrid-precision').textContent = formatPercent(metrics.hybrid.precision);
        document.getElementById('hybrid-recall').textContent = formatPercent(metrics.hybrid.recall);
        document.getElementById('hybrid-f1').textContent = formatPercent(metrics.hybrid.f1_score);
    }

    // Summary metrics
    document.getElementById('ml-total-pnl').textContent = formatCurrency(overall.total_pnl);
    document.getElementById('ml-total-predictions').textContent = overall.total_predictions.toLocaleString();
    document.getElementById('ml-avg-probability').textContent = formatPercent(overall.avg_hybrid_prob);
}

function updateConfusionMatrices(data) {
    const cm = data.confusion_matrix;

    // XGBoost confusion matrix (API uses uppercase TP, TN, FP, FN)
    if (cm.xgboost) {
        document.getElementById('xgboost-tp').textContent = cm.xgboost.TP || 0;
        document.getElementById('xgboost-tn').textContent = cm.xgboost.TN || 0;
        document.getElementById('xgboost-fp').textContent = cm.xgboost.FP || 0;
        document.getElementById('xgboost-fn').textContent = cm.xgboost.FN || 0;
    }

    // Hybrid confusion matrix (API uses uppercase TP, TN, FP, FN)
    if (cm.hybrid) {
        document.getElementById('hybrid-tp').textContent = cm.hybrid.TP || 0;
        document.getElementById('hybrid-tn').textContent = cm.hybrid.TN || 0;
        document.getElementById('hybrid-fp').textContent = cm.hybrid.FP || 0;
        document.getElementById('hybrid-fn').textContent = cm.hybrid.FN || 0;
    }
}

function updateRecommendations(data) {
    // API returns array: [{name: 'TAKE', count: ..., accuracy: ..., total_pnl: ...}, ...]
    const recs = data.recommendations || [];

    recs.forEach(rec => {
        const name = rec.name.toLowerCase();
        const countEl = document.getElementById(`${name}-count`);
        const accEl = document.getElementById(`${name}-accuracy`);
        const pnlEl = document.getElementById(`${name}-pnl`);

        if (countEl) countEl.textContent = rec.count || 0;
        if (accEl) accEl.textContent = formatPercent(rec.accuracy);
        if (pnlEl) pnlEl.textContent = formatCurrency(rec.total_pnl);
    });
}

function updateConfidenceLevels(data) {
    // API returns array: [{name: 'HIGH', count: ..., accuracy: ..., total_pnl: ...}, ...]
    const conf = data.confidence_levels || [];

    conf.forEach(c => {
        const name = c.name.toLowerCase();
        const countEl = document.getElementById(`${name}-count`);
        const accEl = document.getElementById(`${name}-accuracy`);
        const pnlEl = document.getElementById(`${name}-pnl`);

        if (countEl) countEl.textContent = c.count || 0;
        if (accEl) accEl.textContent = formatPercent(c.accuracy);
        if (pnlEl) pnlEl.textContent = formatCurrency(c.total_pnl);
    });
}

function createMLCharts(data) {
    createMLMonthlyAccuracyChart(data.monthly_performance);
    createMLSymbolChart(data.top_symbols);
}

function createMLMonthlyAccuracyChart(monthlyData) {
    const ctx = document.getElementById('mlMonthlyAccuracyChart');
    if (!ctx) return;

    if (mlMonthlyAccuracyChart) {
        mlMonthlyAccuracyChart.destroy();
    }

    if (!monthlyData || monthlyData.length === 0) return;

    const labels = monthlyData.map(d => d.month);
    const accuracies = monthlyData.map(d => d.accuracy);
    const counts = monthlyData.map(d => d.total);  // API uses 'total' not 'count'

    mlMonthlyAccuracyChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    borderColor: 'rgba(99, 102, 241, 1)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Predictions',
                    data: counts,
                    type: 'bar',
                    backgroundColor: 'rgba(168, 85, 247, 0.5)',
                    borderColor: 'rgba(168, 85, 247, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: { color: '#94a3b8' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function createMLSymbolChart(symbolData) {
    const ctx = document.getElementById('mlSymbolChart');
    if (!ctx) return;

    if (mlSymbolChart) {
        mlSymbolChart.destroy();
    }

    if (!symbolData || symbolData.length === 0) return;

    const labels = symbolData.map(d => d.symbol);
    const accuracies = symbolData.map(d => d.accuracy);
    const pnls = symbolData.map(d => d.total_pnl);

    mlSymbolChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                    borderRadius: 4
                },
                {
                    label: 'Total P&L',
                    data: pnls,
                    type: 'line',
                    borderColor: pnls.map(p => p >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'),
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 6,
                    pointBackgroundColor: pnls.map(p => p >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'),
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.label === 'Total P&L') {
                                return 'P&L: ' + formatCurrency(context.parsed.y);
                            }
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45 },
                    grid: { display: false }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function renderPredictionsPage() {
    const tbody = document.getElementById('recentPredictionsBody');
    if (!tbody) return;

    const total = allPredictionsData.length;
    if (total === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-secondary);">No predictions available</td></tr>';
        updatePredictionsPaginationInfo(0, 0, 0);
        buildPredictionsPagination(1, 1);
        return;
    }

    const totalPages = Math.ceil(total / PREDICTIONS_PER_PAGE);
    const startIdx = (predictionsCurrentPage - 1) * PREDICTIONS_PER_PAGE;
    const endIdx = Math.min(startIdx + PREDICTIONS_PER_PAGE, total);
    const pageData = allPredictionsData.slice(startIdx, endIdx);

    let html = '';
    pageData.forEach(pred => {
        // API returns: account, symbol, date, recommendation, confidence, probability, outcome, pnl, correct
        const recClass = pred.recommendation.toLowerCase();
        const confClass = pred.confidence.toLowerCase();
        const outcomeClass = pred.outcome === 'WIN' ? 'win' : 'loss';
        const correctClass = pred.correct ? 'yes' : 'no';
        const correctIcon = pred.correct ? '&#10004;' : '&#10008;';
        const pnlColor = pred.pnl >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
        const displayDate = pred.date ? pred.date.substring(0, 10) : '--';

        html += `
            <tr>
                <td>${displayDate}</td>
                <td>${pred.account}</td>
                <td>${pred.symbol}</td>
                <td><span class="rec-badge ${recClass}">${pred.recommendation}</span></td>
                <td><span class="conf-badge ${confClass}">${pred.confidence}</span></td>
                <td>${formatPercent(pred.probability)}</td>
                <td><span class="outcome-badge ${outcomeClass}">${pred.outcome}</span></td>
                <td style="color: ${pnlColor}">${formatCurrency(pred.pnl)}</td>
                <td><span class="correct-icon ${correctClass}">${correctIcon}</span></td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    updatePredictionsPaginationInfo(startIdx + 1, endIdx, total);
    buildPredictionsPagination(predictionsCurrentPage, totalPages);
}

function updatePredictionsPaginationInfo(from, to, total) {
    const fromEl = document.getElementById('predictions-showing-from');
    const toEl = document.getElementById('predictions-showing-to');
    const totalEl = document.getElementById('predictions-showing-total');
    if (fromEl) fromEl.textContent = from;
    if (toEl) toEl.textContent = to;
    if (totalEl) totalEl.textContent = total;
}

function buildPredictionsPagination(page, totalPages) {
    const container = document.getElementById('predictionsPagination');
    if (!container) return;

    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    html += `<button class="pagination-btn" onclick="goToPredictionsPage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>&#8249; Prev</button>`;

    // Show page numbers
    const maxVisible = 5;
    let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        html += `<button class="pagination-btn" onclick="goToPredictionsPage(1)">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="pagination-btn ${i === page ? 'active' : ''}" onclick="goToPredictionsPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
        html += `<button class="pagination-btn" onclick="goToPredictionsPage(${totalPages})">${totalPages}</button>`;
    }

    html += `<button class="pagination-btn" onclick="goToPredictionsPage(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>Next &#8250;</button>`;
    container.innerHTML = html;
}

function goToPredictionsPage(page) {
    const totalPages = Math.ceil(allPredictionsData.length / PREDICTIONS_PER_PAGE);
    if (page < 1 || page > totalPages) return;
    predictionsCurrentPage = page;
    renderPredictionsPage();
}

// =============================================================================
// TAB 4: TRADE EXPLORER
// =============================================================================
let tradesData = null;
let tradesFilters = null;
let currentPage = 1;
let tradeSymbolChart = null;

async function loadTradesData() {
    try {
        // Load filters, stats, and initial trades in parallel
        const [filtersRes, statsRes, tradesRes] = await Promise.all([
            fetch('/api/trades/filters'),
            fetch('/api/trades/stats'),
            fetch('/api/trades?page=1&per_page=50')
        ]);

        tradesFilters = await filtersRes.json();
        const stats = await statsRes.json();
        tradesData = await tradesRes.json();

        populateTradeFilters(tradesFilters);
        updateTradeStats(stats);
        createTradeCharts(stats);
        populateTradesTable(tradesData);

    } catch (error) {
        console.error('Error loading trades data:', error);
    }
}

function populateTradeFilters(filters) {
    // Populate account dropdown
    const accountSelect = document.getElementById('filter-account');
    if (accountSelect && filters.accounts) {
        filters.accounts.forEach(acc => {
            const option = document.createElement('option');
            option.value = acc;
            option.textContent = acc;
            accountSelect.appendChild(option);
        });
    }

    // Populate symbol dropdown
    const symbolSelect = document.getElementById('filter-symbol');
    if (symbolSelect && filters.symbols) {
        filters.symbols.forEach(sym => {
            const option = document.createElement('option');
            option.value = sym;
            option.textContent = sym;
            symbolSelect.appendChild(option);
        });
    }

    // Set date range
    if (filters.date_range) {
        const fromDate = document.getElementById('filter-date-from');
        const toDate = document.getElementById('filter-date-to');
        if (fromDate && filters.date_range.min) fromDate.min = filters.date_range.min;
        if (toDate && filters.date_range.max) toDate.max = filters.date_range.max;
    }
}

function updateTradeStats(stats) {
    const overall = stats.overall;
    document.getElementById('trades-total').textContent = overall.total_trades.toLocaleString();
    document.getElementById('trades-winning').textContent = overall.winning_trades.toLocaleString();
    document.getElementById('trades-losing').textContent = overall.losing_trades.toLocaleString();
    document.getElementById('trades-winrate').textContent = overall.win_rate + '%';
    document.getElementById('trades-total-pnl').textContent = formatCurrency(overall.total_pnl);
    document.getElementById('trades-avg-pnl').textContent = formatCurrency(overall.avg_pnl);
}

function createTradeCharts(stats) {
    createTradePnlDistChart(stats.pnl_distribution);
    createTradeSymbolChart(stats.symbol_performance);
}

function createTradePnlDistChart(data) {
    const ctx = document.getElementById('tradePnlDistributionChart');
    if (!ctx) return;

    if (tradePnlDistChart) {
        tradePnlDistChart.destroy();
    }

    if (!data || data.length === 0) {
        console.log('No P&L distribution data');
        return;
    }

    // Order the ranges properly - negative ranges first, then positive
    const negativeRanges = ['< -500', '-500 to -200', '-200 to -100', '-100 to 0'];
    const positiveRanges = ['0 to 100', '100 to 200', '200 to 500', '> 500'];
    const rangeOrder = [...negativeRanges, ...positiveRanges];

    const sortedData = rangeOrder.map(r => data.find(d => d.range === r) || { range: r, count: 0 });

    const labels = sortedData.map(d => d.range);
    const counts = sortedData.map(d => d.count);

    // Explicit color mapping - red for negative, green for positive
    const colors = sortedData.map((d, i) => {
        return i < 4 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)';
    });
    const borderColors = sortedData.map((d, i) => {
        return i < 4 ? 'rgba(239, 68, 68, 1)' : 'rgba(16, 185, 129, 1)';
    });

    tradePnlDistChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of Trades',
                data: counts,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                }
            }
        }
    });
}

function createTradeSymbolChart(data) {
    const ctx = document.getElementById('tradeSymbolChart');
    if (!ctx) return;

    if (tradeSymbolChart) {
        tradeSymbolChart.destroy();
    }

    const labels = data.map(d => d.symbol);
    const winRates = data.map(d => d.win_rate);
    const pnls = data.map(d => d.total_pnl);

    tradeSymbolChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Win Rate (%)',
                    data: winRates,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Total P&L',
                    data: pnls,
                    type: 'line',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 5,
                    pointBackgroundColor: pnls.map(p => p >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'),
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45 },
                    grid: { display: false }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: v => v + '%'
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: {
                        color: '#94a3b8',
                        callback: v => '$' + v.toLocaleString()
                    },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function populateTradesTable(data) {
    const tbody = document.getElementById('tradesTableBody');
    if (!tbody) return;

    const trades = data.trades || [];
    const pagination = data.pagination || {};

    // Update pagination info
    document.getElementById('showing-from').textContent = trades.length > 0 ? ((pagination.page - 1) * pagination.per_page + 1) : 0;
    document.getElementById('showing-to').textContent = Math.min(pagination.page * pagination.per_page, pagination.total);
    document.getElementById('showing-total').textContent = pagination.total;
    document.getElementById('filtered-trade-count').textContent = `${pagination.total} trades`;

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-secondary);">No trades found</td></tr>';
        return;
    }

    let html = '';
    trades.forEach(trade => {
        const recClass = trade.recommendation.toLowerCase();
        const confClass = trade.confidence.toLowerCase();
        const outcomeClass = trade.outcome === 'WIN' ? 'win' : 'loss';
        const correctClass = trade.hybrid_correct ? 'yes' : 'no';
        const correctIcon = trade.hybrid_correct ? '&#10004;' : '&#10008;';
        const pnlColor = trade.pnl >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
        const displayDate = trade.date ? trade.date.substring(0, 10) : '--';

        html += `
            <tr>
                <td>${displayDate}</td>
                <td>${trade.account}</td>
                <td>${trade.symbol}</td>
                <td><span class="rec-badge ${recClass}">${trade.recommendation}</span></td>
                <td><span class="conf-badge ${confClass}">${trade.confidence}</span></td>
                <td>${formatPercent(trade.hybrid_prob)}</td>
                <td>${formatPercent(trade.xgboost_prob)}</td>
                <td><span class="outcome-badge ${outcomeClass}">${trade.outcome}</span></td>
                <td style="color: ${pnlColor}">${formatCurrency(trade.pnl)}</td>
                <td><span class="correct-icon ${correctClass}">${correctIcon}</span></td>
            </tr>
        `;
    });

    tbody.innerHTML = html;

    // Build pagination
    buildPagination(pagination);
}

function buildPagination(pagination) {
    const container = document.getElementById('tradesPagination');
    if (!container) return;

    const { page, total_pages } = pagination;
    let html = '';

    // Previous button
    html += `<button class="pagination-btn" onclick="goToPage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>&#8249; Prev</button>`;

    // Page numbers
    const maxVisible = 5;
    let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
    let endPage = Math.min(total_pages, startPage + maxVisible - 1);

    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        html += `<button class="pagination-btn" onclick="goToPage(1)">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="pagination-btn ${i === page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < total_pages) {
        if (endPage < total_pages - 1) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
        html += `<button class="pagination-btn" onclick="goToPage(${total_pages})">${total_pages}</button>`;
    }

    // Next button
    html += `<button class="pagination-btn" onclick="goToPage(${page + 1})" ${page >= total_pages ? 'disabled' : ''}>Next &#8250;</button>`;

    container.innerHTML = html;
}

async function goToPage(page) {
    if (page < 1) return;
    currentPage = page;
    await applyTradeFilters(false);  // Don't reset page when navigating
}

async function applyTradeFilters(resetPage = true) {
    // Reset to page 1 when applying new filters (not when navigating)
    if (resetPage) {
        currentPage = 1;
    }

    const account = document.getElementById('filter-account').value;
    const symbol = document.getElementById('filter-symbol').value;
    const recommendation = document.getElementById('filter-recommendation').value;
    const confidence = document.getElementById('filter-confidence').value;
    const outcome = document.getElementById('filter-outcome').value;
    const dateFrom = document.getElementById('filter-date-from').value;
    const dateTo = document.getElementById('filter-date-to').value;
    const sortBy = document.getElementById('sort-by').value;
    const sortOrder = document.getElementById('sort-order').value;

    // Build filter params for both trades and stats
    const filterParams = new URLSearchParams();
    if (account) filterParams.append('account', account);
    if (symbol) filterParams.append('symbol', symbol);
    if (recommendation) filterParams.append('recommendation', recommendation);
    if (confidence) filterParams.append('confidence', confidence);
    if (outcome) filterParams.append('outcome', outcome);
    if (dateFrom) filterParams.append('date_from', dateFrom);
    if (dateTo) filterParams.append('date_to', dateTo);

    // Build params for trades endpoint (includes pagination and sorting)
    const tradesParams = new URLSearchParams(filterParams);
    tradesParams.append('page', currentPage);
    tradesParams.append('per_page', 50);
    tradesParams.append('sort_by', sortBy);
    tradesParams.append('sort_order', sortOrder);

    try {
        // Fetch both trades and stats with filters
        const [tradesResponse, statsResponse] = await Promise.all([
            fetch(`/api/trades?${tradesParams.toString()}`),
            fetch(`/api/trades/stats?${filterParams.toString()}`)
        ]);

        tradesData = await tradesResponse.json();
        const stats = await statsResponse.json();

        // Update table
        populateTradesTable(tradesData);

        // Update stats cards
        updateTradeStats(stats);

        // Update charts with filtered data
        createTradeCharts(stats);

    } catch (error) {
        console.error('Error applying filters:', error);
    }
}

function resetTradeFilters() {
    document.getElementById('filter-account').value = '';
    document.getElementById('filter-symbol').value = '';
    document.getElementById('filter-recommendation').value = '';
    document.getElementById('filter-confidence').value = '';
    document.getElementById('filter-outcome').value = '';
    document.getElementById('filter-date-from').value = '';
    document.getElementById('filter-date-to').value = '';
    document.getElementById('sort-by').value = 'entry_ts';
    document.getElementById('sort-order').value = 'DESC';
    currentPage = 1;
    applyTradeFilters();
}

// =============================================================================
// TAB 5: TRADE SCREENER
// =============================================================================
let screenerData = null;
let allSymbolsData = [];
let allAccountsData = [];
let opportunitiesCurrentPage = 1;
let avoidCurrentPage = 1;
let accountsScreenerCurrentPage = 1;
const SCREENER_PER_PAGE = 10;

async function loadScreenerData() {
    try {
        // Load all screener data in parallel
        const [symbolsRes, accountsRes, insightsRes] = await Promise.all([
            fetch('/api/screener/symbols'),
            fetch('/api/screener/accounts'),
            fetch('/api/screener/insights')
        ]);

        allSymbolsData = await symbolsRes.json();
        allAccountsData = await accountsRes.json();
        const insightsData = await insightsRes.json();

        screenerData = {
            symbols: allSymbolsData,
            accounts: allAccountsData,
            insights: insightsData
        };

        // Update UI
        updateScreenerInsights(insightsData);
        updateScreenerSummary(insightsData);

        // Initialize pagination for screener tables
        opportunitiesCurrentPage = 1;
        avoidCurrentPage = 1;
        accountsScreenerCurrentPage = 1;
        renderOpportunitiesPage();
        renderAvoidPage();
        renderAccountsPage();

    } catch (error) {
        console.error('Error loading screener data:', error);
    }
}

function updateScreenerInsights(data) {
    const container = document.getElementById('screenerInsights');
    if (!container) return;

    const insights = data.insights || [];

    if (insights.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No insights available</p>';
        return;
    }

    // Map API types to CSS classes
    const typeMapping = {
        'success': 'opportunity',
        'opportunity': 'opportunity',
        'warning': 'warning',
        'info': 'info',
        'danger': 'danger'
    };

    let html = '';
    insights.forEach(insight => {
        const cssClass = typeMapping[insight.type] || 'info';
        html += `
            <div class="insight-card ${cssClass}">
                <span class="insight-icon">${insight.icon}</span>
                <div class="insight-content">
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-description">${insight.description}</div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function updateScreenerSummary(data) {
    const summary = data.summary || {};

    // Best symbol
    document.getElementById('screener-best-symbol').textContent = summary.best_symbol || '--';

    // Worst symbol
    document.getElementById('screener-worst-symbol').textContent = summary.worst_symbol || '--';

    // High confidence accuracy
    const highConfAcc = summary.high_conf_accuracy || 0;
    document.getElementById('screener-high-conf-accuracy').textContent = formatPercent(highConfAcc);

    // Count HOT signals
    const hotCount = allSymbolsData.filter(s => s.signal === 'HOT').length;
    document.getElementById('screener-hot-count').textContent = hotCount;
}

function getOpportunitiesData() {
    // Get all opportunities (highest ML accuracy with decent trade count)
    return allSymbolsData
        .filter(s => s.total_trades >= 5)  // At least 5 trades
        .sort((a, b) => b.ml_accuracy - a.ml_accuracy);
}

function renderOpportunitiesPage() {
    const tbody = document.getElementById('screenerOpportunitiesBody');
    if (!tbody) return;

    const allOpportunities = getOpportunitiesData();
    const total = allOpportunities.length;

    if (total === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-secondary);">No opportunities found</td></tr>';
        updateScreenerPaginationInfo('opportunities', 0, 0, 0);
        buildScreenerPagination('opportunities', 1, 1);
        return;
    }

    const totalPages = Math.ceil(total / SCREENER_PER_PAGE);
    const startIdx = (opportunitiesCurrentPage - 1) * SCREENER_PER_PAGE;
    const endIdx = Math.min(startIdx + SCREENER_PER_PAGE, total);
    const pageData = allOpportunities.slice(startIdx, endIdx);

    let html = '';
    pageData.forEach(sym => {
        const signalClass = sym.signal.toLowerCase();
        const pnlClass = sym.total_pnl >= 0 ? 'positive-value' : 'negative-value';
        const avgPnlClass = sym.avg_pnl >= 0 ? 'positive-value' : 'negative-value';

        html += `
            <tr>
                <td class="symbol-name">${sym.symbol}</td>
                <td>
                    <span class="signal-cell">
                        <span class="signal-indicator ${signalClass}"></span>
                        <span class="signal-badge ${signalClass}">${sym.signal}</span>
                    </span>
                </td>
                <td>${sym.total_trades}</td>
                <td>${formatPercent(sym.win_rate)}</td>
                <td class="highlight-value">${formatPercent(sym.ml_accuracy)}</td>
                <td>${formatPercent(sym.take_accuracy || 0)}</td>
                <td>${formatPercent(sym.high_conf_accuracy || 0)}</td>
                <td class="${pnlClass}">${formatCurrency(sym.total_pnl)}</td>
                <td class="${avgPnlClass}">${formatCurrency(sym.avg_pnl)}</td>
                <td>${formatNumber(sym.risk_reward || 0, 2)}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    updateScreenerPaginationInfo('opportunities', startIdx + 1, endIdx, total);
    buildScreenerPagination('opportunities', opportunitiesCurrentPage, totalPages);
}

function getAvoidData() {
    // Get symbols to avoid (lowest ML accuracy)
    return allSymbolsData
        .filter(s => s.total_trades >= 5 && (s.signal === 'WEAK' || s.signal === 'NEUTRAL'))
        .sort((a, b) => a.ml_accuracy - b.ml_accuracy);
}

function renderAvoidPage() {
    const tbody = document.getElementById('screenerAvoidBody');
    if (!tbody) return;

    const allAvoid = getAvoidData();
    const total = allAvoid.length;

    if (total === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">No symbols to avoid</td></tr>';
        updateScreenerPaginationInfo('avoid', 0, 0, 0);
        buildScreenerPagination('avoid', 1, 1);
        return;
    }

    const totalPages = Math.ceil(total / SCREENER_PER_PAGE);
    const startIdx = (avoidCurrentPage - 1) * SCREENER_PER_PAGE;
    const endIdx = Math.min(startIdx + SCREENER_PER_PAGE, total);
    const pageData = allAvoid.slice(startIdx, endIdx);

    let html = '';
    pageData.forEach(sym => {
        const signalClass = sym.signal.toLowerCase();
        const pnlClass = sym.total_pnl >= 0 ? 'positive-value' : 'negative-value';

        // Determine reason to avoid
        let reason = '';
        if (sym.ml_accuracy < 50) {
            reason = 'Low ML accuracy';
        } else if (sym.win_rate < 40) {
            reason = 'Low win rate';
        } else if (sym.total_pnl < 0) {
            reason = 'Negative P&L';
        } else {
            reason = 'Weak signal';
        }

        html += `
            <tr>
                <td class="symbol-name">${sym.symbol}</td>
                <td>
                    <span class="signal-cell">
                        <span class="signal-indicator ${signalClass}"></span>
                        <span class="signal-badge ${signalClass}">${sym.signal}</span>
                    </span>
                </td>
                <td>${sym.total_trades}</td>
                <td>${formatPercent(sym.win_rate)}</td>
                <td>${formatPercent(sym.ml_accuracy)}</td>
                <td>${formatPercent(sym.take_accuracy || 0)}</td>
                <td class="${pnlClass}">${formatCurrency(sym.total_pnl)}</td>
                <td><span class="avoid-reason">${reason}</span></td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    updateScreenerPaginationInfo('avoid', startIdx + 1, endIdx, total);
    buildScreenerPagination('avoid', avoidCurrentPage, totalPages);
}

function getAccountsData() {
    // Sort by total P&L for ranking
    return [...allAccountsData].sort((a, b) => b.total_pnl - a.total_pnl);
}

function renderAccountsPage() {
    const tbody = document.getElementById('screenerAccountsBody');
    if (!tbody) return;

    const sortedAccounts = getAccountsData();
    const total = sortedAccounts.length;

    if (total === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-secondary);">No accounts found</td></tr>';
        updateScreenerPaginationInfo('accounts', 0, 0, 0);
        buildScreenerPagination('accounts', 1, 1);
        return;
    }

    const totalPages = Math.ceil(total / SCREENER_PER_PAGE);
    const startIdx = (accountsScreenerCurrentPage - 1) * SCREENER_PER_PAGE;
    const endIdx = Math.min(startIdx + SCREENER_PER_PAGE, total);
    const pageData = sortedAccounts.slice(startIdx, endIdx);

    let html = '';
    pageData.forEach((acc, idx) => {
        const rank = startIdx + idx + 1;  // Adjust rank for pagination
        let rankClass = 'default';
        if (rank === 1) rankClass = 'gold';
        else if (rank === 2) rankClass = 'silver';
        else if (rank === 3) rankClass = 'bronze';

        const pnlClass = acc.total_pnl >= 0 ? 'positive-value' : 'negative-value';
        const beatClass = acc.beat_market ? 'yes' : 'no';
        const outperformClass = acc.outperformance >= 0 ? 'positive-value' : 'negative-value';

        html += `
            <tr>
                <td><span class="rank-badge ${rankClass}">${rank}</span></td>
                <td class="symbol-name">${acc.account_id}</td>
                <td>${acc.total_trades}</td>
                <td>${formatPercent(acc.win_rate)}</td>
                <td>${formatPercent(acc.ml_accuracy)}</td>
                <td class="${pnlClass}">${formatCurrency(acc.total_pnl)}</td>
                <td>${formatNumber(acc.sharpe_ratio || 0, 2)}</td>
                <td><span class="beat-market-badge ${beatClass}">${acc.beat_market ? 'Yes' : 'No'}</span></td>
                <td class="${outperformClass}">${formatPercent(acc.outperformance || 0)}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    updateScreenerPaginationInfo('accounts', startIdx + 1, endIdx, total);
    buildScreenerPagination('accounts', accountsScreenerCurrentPage, totalPages);
}

// Screener Pagination Helper Functions
function updateScreenerPaginationInfo(type, from, to, total) {
    const fromEl = document.getElementById(`${type}-showing-from`);
    const toEl = document.getElementById(`${type}-showing-to`);
    const totalEl = document.getElementById(`${type}-showing-total`);
    if (fromEl) fromEl.textContent = from;
    if (toEl) toEl.textContent = to;
    if (totalEl) totalEl.textContent = total;
}

function buildScreenerPagination(type, page, totalPages) {
    const container = document.getElementById(`${type}Pagination`);
    if (!container) return;

    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    html += `<button class="pagination-btn" onclick="goToScreenerPage('${type}', ${page - 1})" ${page <= 1 ? 'disabled' : ''}>&#8249; Prev</button>`;

    // Show page numbers
    const maxVisible = 5;
    let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        html += `<button class="pagination-btn" onclick="goToScreenerPage('${type}', 1)">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="pagination-btn ${i === page ? 'active' : ''}" onclick="goToScreenerPage('${type}', ${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
        html += `<button class="pagination-btn" onclick="goToScreenerPage('${type}', ${totalPages})">${totalPages}</button>`;
    }

    html += `<button class="pagination-btn" onclick="goToScreenerPage('${type}', ${page + 1})" ${page >= totalPages ? 'disabled' : ''}>Next &#8250;</button>`;
    container.innerHTML = html;
}

function goToScreenerPage(type, page) {
    if (type === 'opportunities') {
        const totalPages = Math.ceil(getOpportunitiesData().length / SCREENER_PER_PAGE);
        if (page < 1 || page > totalPages) return;
        opportunitiesCurrentPage = page;
        renderOpportunitiesPage();
    } else if (type === 'avoid') {
        const totalPages = Math.ceil(getAvoidData().length / SCREENER_PER_PAGE);
        if (page < 1 || page > totalPages) return;
        avoidCurrentPage = page;
        renderAvoidPage();
    } else if (type === 'accounts') {
        const totalPages = Math.ceil(getAccountsData().length / SCREENER_PER_PAGE);
        if (page < 1 || page > totalPages) return;
        accountsScreenerCurrentPage = page;
        renderAccountsPage();
    }
}

function applyScreenerFilters() {
    const signalFilter = document.getElementById('screener-signal').value;
    const minTrades = parseInt(document.getElementById('screener-min-trades').value) || 0;
    const minAccuracy = parseFloat(document.getElementById('screener-min-accuracy').value) || 0;

    // Filter symbols
    let filteredSymbols = allSymbolsData.filter(sym => {
        if (signalFilter && sym.signal !== signalFilter) return false;
        if (sym.total_trades < minTrades) return false;
        if (sym.ml_accuracy < minAccuracy) return false;
        return true;
    });

    // Update tables with filtered data
    populateOpportunitiesTable(filteredSymbols);
    populateAvoidTable(filteredSymbols);
}

function resetScreenerFilters() {
    document.getElementById('screener-signal').value = '';
    document.getElementById('screener-min-trades').value = '0';
    document.getElementById('screener-min-accuracy').value = '0';

    // Reset with original data
    populateOpportunitiesTable(allSymbolsData);
    populateAvoidTable(allSymbolsData);
}

// =============================================================================
// TAB 6: MARKET DATA
// =============================================================================
let marketData = null;
let marketPriceChart = null;
let currentMarketSymbol = null;
let currentSectorFilter = 'all';

// Sector mappings for filtering market data
const SECTOR_MAPPINGS = {
    tech: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL'],
    finance: ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'SCHW'],
    consumer: ['WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'DIS'],
    healthcare: ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT'],
    energy: ['XOM', 'CVX', 'COP', 'SLB'],
    etfs: ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'TLT', 'HYG'],
    crypto: ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
};

// Get sector for a symbol
function getSymbolSector(symbol) {
    for (const [sector, symbols] of Object.entries(SECTOR_MAPPINGS)) {
        if (symbols.includes(symbol)) {
            return sector;
        }
    }
    return 'other';
}

async function loadMarketData() {
    try {
        // Load market summary and symbols
        const [summaryRes, symbolsRes] = await Promise.all([
            fetch('/api/market/summary'),
            fetch('/api/market/symbols')
        ]);

        const summary = await summaryRes.json();
        const symbols = await symbolsRes.json();

        marketData = { summary, symbols };

        // Initialize sector filter tabs
        initSectorFilterTabs();

        // Populate symbol cards (filtered by current sector)
        populateMarketSummaryGrid(summary);

        // Populate symbol dropdown
        populateMarketSymbolSelect(symbols);

    } catch (error) {
        console.error('Error loading market data:', error);
    }
}

function initSectorFilterTabs() {
    const tabs = document.querySelectorAll('.sector-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Update active state
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update filter and refresh grid
            currentSectorFilter = tab.dataset.sector;
            if (marketData) {
                populateMarketSummaryGrid(marketData.summary);
            }
        });
    });
}

function populateMarketSummaryGrid(summary) {
    const grid = document.getElementById('marketSummaryGrid');
    if (!grid) return;

    // Filter by sector
    let filteredSummary = summary;
    if (currentSectorFilter !== 'all') {
        const sectorSymbols = SECTOR_MAPPINGS[currentSectorFilter] || [];
        filteredSummary = summary.filter(item => sectorSymbols.includes(item.symbol));
    }

    let html = '';

    if (filteredSummary.length === 0) {
        html = '<div class="no-data-message">No instruments found in this category</div>';
    } else {
        filteredSummary.forEach(item => {
            const changeClass = item.change_pct >= 0 ? 'positive' : 'negative';
            const changeSign = item.change_pct >= 0 ? '+' : '';
            const sector = getSymbolSector(item.symbol);
            const sectorClass = sector !== 'other' ? `sector-${sector}` : '';

            html += `
                <div class="market-symbol-card ${sectorClass}" onclick="selectMarketSymbol('${item.symbol}')">
                    <div class="market-symbol-name">${item.symbol}</div>
                    <div class="market-symbol-price">${formatCurrency(item.price)}</div>
                    <div class="market-symbol-change ${changeClass}">
                        ${changeSign}${item.change_pct?.toFixed(2) || '0.00'}%
                    </div>
                </div>
            `;
        });
    }

    grid.innerHTML = html;
}

function populateMarketSymbolSelect(symbols) {
    const select = document.getElementById('marketSymbolSelect');
    if (!select) return;

    select.innerHTML = '<option value="">Select a symbol...</option>';

    symbols.forEach(sym => {
        const option = document.createElement('option');
        option.value = sym.symbol;
        option.textContent = `${sym.symbol} (${sym.price_count} prices)`;
        select.appendChild(option);
    });

    // Add change listener
    select.addEventListener('change', (e) => {
        if (e.target.value) {
            selectMarketSymbol(e.target.value);
        }
    });

    // Add timeframe change listener
    const timeframe = document.getElementById('marketTimeframe');
    if (timeframe) {
        timeframe.addEventListener('change', () => {
            if (currentMarketSymbol) {
                loadSymbolPrices(currentMarketSymbol);
            }
        });
    }
}

async function selectMarketSymbol(symbol) {
    currentMarketSymbol = symbol;

    // Update select dropdown
    const select = document.getElementById('marketSymbolSelect');
    if (select) select.value = symbol;

    // Update card active states
    document.querySelectorAll('.market-symbol-card').forEach(card => {
        card.classList.remove('active');
        if (card.querySelector('.market-symbol-name')?.textContent === symbol) {
            card.classList.add('active');
        }
    });

    // Load price data
    await loadSymbolPrices(symbol);

    // Show stats section
    document.getElementById('marketStatsSection').style.display = 'block';
}

async function loadSymbolPrices(symbol) {
    try {
        const days = document.getElementById('marketTimeframe')?.value || 365;
        const response = await fetch(`/api/market/prices/${symbol}?days=${days}`);
        const data = await response.json();

        if (data.prices && data.prices.length > 0) {
            createMarketPriceChart(data.prices, symbol);
            updateMarketStats(data.prices);
        }

    } catch (error) {
        console.error('Error loading symbol prices:', error);
    }
}

function createMarketPriceChart(prices, symbol) {
    const ctx = document.getElementById('marketPriceChart');
    if (!ctx) return;

    if (marketPriceChart) {
        marketPriceChart.destroy();
    }

    // Update chart title
    document.getElementById('marketChartTitle').textContent = `${symbol} Price History`;

    const labels = prices.map(p => new Date(p.date).toLocaleDateString());
    const closeData = prices.map(p => p.close);
    const highData = prices.map(p => p.high);
    const lowData = prices.map(p => p.low);

    marketPriceChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Close',
                    data: closeData,
                    borderColor: 'rgba(99, 102, 241, 1)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: 'High',
                    data: highData,
                    borderColor: 'rgba(16, 185, 129, 0.5)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    borderDash: [5, 5]
                },
                {
                    label: 'Low',
                    data: lowData,
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 12 },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    },
                    grid: { color: 'rgba(71, 85, 105, 0.3)' }
                }
            }
        }
    });
}

function updateMarketStats(prices) {
    if (!prices || prices.length === 0) return;

    const latest = prices[prices.length - 1];
    const previous = prices.length > 1 ? prices[prices.length - 2] : latest;

    // Current price
    document.getElementById('market-current-price').textContent = formatCurrency(latest.close);

    // Daily change
    const change = latest.close - previous.close;
    const changePct = (change / previous.close) * 100;
    const changeEl = document.getElementById('market-daily-change');
    changeEl.textContent = `${change >= 0 ? '+' : ''}${formatCurrency(change)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)`;
    changeEl.className = 'market-stat-value ' + (change >= 0 ? 'positive' : 'negative');

    // 52W High/Low (using available data)
    const highs = prices.map(p => p.high);
    const lows = prices.map(p => p.low);
    document.getElementById('market-52w-high').textContent = formatCurrency(Math.max(...highs));
    document.getElementById('market-52w-low').textContent = formatCurrency(Math.min(...lows));

    // Avg Volume
    const volumes = prices.map(p => p.volume || 0);
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    document.getElementById('market-avg-volume').textContent = formatLargeNumber(avgVolume);

    // Volatility (standard deviation of daily returns)
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i].close - prices[i-1].close) / prices[i-1].close);
    }
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252) * 100; // Annualized
    document.getElementById('market-volatility').textContent = volatility.toFixed(2) + '%';
}

function formatLargeNumber(num) {
    if (num >= 1000000000) return (num / 1000000000).toFixed(2) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toFixed(0);
}

// =============================================================================
// TAB 7: DATABASE EXPLORER
// =============================================================================

let databaseTables = [];

async function loadDatabaseData() {
    try {
        // Load database info and tables
        const [infoRes, tablesRes] = await Promise.all([
            fetch('/api/database/info'),
            fetch('/api/database/tables')
        ]);

        const info = await infoRes.json();
        const tables = await tablesRes.json();

        databaseTables = tables;

        // Update stats
        document.getElementById('db-table-count').textContent = info.table_count;
        document.getElementById('db-total-rows').textContent = formatLargeNumber(info.total_rows);
        document.getElementById('db-size').textContent = info.db_size_mb + ' MB';
        document.getElementById('db-index-count').textContent = info.index_count;

        // Populate table list
        populateTableList(tables);

        // Render ER diagram
        renderERDiagram();

        // Initialize Mermaid
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                themeVariables: {
                    primaryColor: '#6366f1',
                    primaryTextColor: '#f1f5f9',
                    primaryBorderColor: '#6366f1',
                    lineColor: '#64748b',
                    secondaryColor: '#1e293b',
                    tertiaryColor: '#0f172a'
                }
            });
        }

    } catch (error) {
        console.error('Error loading database data:', error);
    }
}

function populateTableList(tables) {
    const listContainer = document.getElementById('tableList');
    if (!listContainer) return;

    // Filter out tables with 0 rows
    const nonEmptyTables = tables.filter(table => table.row_count > 0);

    let html = '';
    nonEmptyTables.forEach(table => {
        html += `
            <div class="table-list-item" onclick="loadTableSchema('${table.name}')">
                <span class="table-icon">&#128451;</span>
                <span class="table-name">${table.name}</span>
                <span class="table-row-count">${formatLargeNumber(table.row_count)} rows</span>
            </div>
        `;
    });

    listContainer.innerHTML = html;
}

async function loadTableSchema(tableName) {
    try {
        // Highlight selected table
        document.querySelectorAll('.table-list-item').forEach(item => {
            item.classList.remove('active');
            if (item.querySelector('.table-name').textContent === tableName) {
                item.classList.add('active');
            }
        });

        const response = await fetch(`/api/database/schema/${tableName}`);
        const data = await response.json();

        displayTableDetails(data);

    } catch (error) {
        console.error('Error loading table schema:', error);
    }
}

function displayTableDetails(data) {
    const container = document.getElementById('tableDetails');
    if (!container) return;

    // Build columns table
    let columnsHtml = `
        <div class="table-details-header">
            <h4>${data.table_name}</h4>
            <span class="row-count-badge">${formatLargeNumber(data.row_count)} rows</span>
        </div>
        <div class="schema-columns">
            <h5>Columns</h5>
            <table class="schema-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Column Name</th>
                        <th>Type</th>
                        <th>Nullable</th>
                        <th>Key</th>
                    </tr>
                </thead>
                <tbody>
    `;

    data.columns.forEach(col => {
        const isPK = col.pk === 1;
        const isFK = data.foreign_keys.some(fk => fk.from === col.name);
        let keyBadge = '';
        if (isPK) keyBadge = '<span class="key-badge pk">PK</span>';
        else if (isFK) keyBadge = '<span class="key-badge fk">FK</span>';

        columnsHtml += `
            <tr>
                <td>${col.cid}</td>
                <td><code>${col.name}</code></td>
                <td>${col.type || 'TEXT'}</td>
                <td>${col.notnull ? 'NO' : 'YES'}</td>
                <td>${keyBadge}</td>
            </tr>
        `;
    });

    columnsHtml += '</tbody></table></div>';

    // Foreign keys section
    if (data.foreign_keys.length > 0) {
        columnsHtml += `
            <div class="schema-fks">
                <h5>Foreign Keys</h5>
                <ul>
        `;
        data.foreign_keys.forEach(fk => {
            columnsHtml += `<li><code>${fk.from}</code> &#8594; <code>${fk.table}.${fk.to}</code></li>`;
        });
        columnsHtml += '</ul></div>';
    }

    // Sample data section
    if (data.sample_data.length > 0) {
        const columns = Object.keys(data.sample_data[0]);
        columnsHtml += `
            <div class="schema-sample">
                <h5>Sample Data (First 5 rows)</h5>
                <div class="sample-table-wrapper">
                    <table class="sample-table">
                        <thead>
                            <tr>
                                ${columns.map(c => `<th>${c}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
        `;
        data.sample_data.forEach(row => {
            columnsHtml += '<tr>';
            columns.forEach(col => {
                let val = row[col];
                if (val === null) val = '<span class="null-value">NULL</span>';
                else if (typeof val === 'string' && val.length > 30) val = val.substring(0, 30) + '...';
                columnsHtml += `<td>${val}</td>`;
            });
            columnsHtml += '</tr>';
        });
        columnsHtml += '</tbody></table></div></div>';
    }

    container.innerHTML = columnsHtml;
}

function renderERDiagram() {
    const container = document.getElementById('erDiagram');
    if (!container) {
        console.log('ER diagram container not found');
        return;
    }

    if (typeof mermaid === 'undefined') {
        console.log('Mermaid not loaded');
        container.innerHTML = '<p style="color: var(--text-secondary);">Loading diagram...</p>';
        return;
    }

    // Simplified ER diagram - core tables only for better display
    const erDiagramCode = `erDiagram
    BROKERS {
        int broker_id PK
        string name
        string broker_type
    }
    ACCOUNTS {
        string account_id PK
        int broker_id FK
        string currency
        date opened_at
    }
    INSTRUMENTS {
        int instrument_id PK
        string symbol
        string asset_class
    }
    ORDERS {
        int order_id PK
        string account_id FK
        int instrument_id FK
        string side
    }
    FILLS {
        int fill_id PK
        int order_id FK
        datetime trade_ts
        real qty
        real price
    }
    ML_PREDICTIONS {
        int prediction_id PK
        int fill_id FK
        string recommendation
        real confidence
    }
    ACCOUNT_METRICS {
        string account_id FK
        real net_pnl
        real win_rate
        real sharpe_ratio
    }
    PRICES {
        int instrument_id FK
        datetime ts
        real px_close
    }

    BROKERS ||--o{ ACCOUNTS : has
    ACCOUNTS ||--o{ ORDERS : places
    ACCOUNTS ||--o{ ACCOUNT_METRICS : has
    INSTRUMENTS ||--o{ ORDERS : for
    INSTRUMENTS ||--o{ PRICES : has
    ORDERS ||--o{ FILLS : filled
    FILLS ||--o{ ML_PREDICTIONS : predicts`;

    // Clear and set up the container
    container.innerHTML = '';

    // Create div with mermaid class
    const diagramDiv = document.createElement('div');
    diagramDiv.className = 'mermaid';
    diagramDiv.textContent = erDiagramCode;
    container.appendChild(diagramDiv);

    // Initialize mermaid with dark theme
    mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        er: {
            layoutDirection: 'TB',
            minEntityWidth: 100,
            minEntityHeight: 75,
            entityPadding: 15
        },
        themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#f1f5f9',
            primaryBorderColor: '#6366f1',
            lineColor: '#64748b',
            secondaryColor: '#1e293b',
            tertiaryColor: '#0f172a',
            background: '#0f172a'
        }
    });

    // Render the diagram
    mermaid.run({
        nodes: [diagramDiv]
    }).catch(err => {
        console.error('Mermaid render error:', err);
        container.innerHTML = '<p style="color: var(--danger-color);">Error rendering diagram</p>';
    });
}

// SQL Query functions
const exampleQueries = {
    accounts: 'SELECT * FROM accounts LIMIT 10',
    top_trades: `SELECT
    mp.symbol,
    mp.entry_ts,
    mp.actual_pnl,
    mp.recommendation,
    mp.confidence_level
FROM ml_predictions mp
ORDER BY mp.actual_pnl DESC
LIMIT 10`,
    symbol_stats: `SELECT
    symbol,
    COUNT(*) as total_trades,
    ROUND(SUM(actual_pnl), 2) as total_pnl,
    ROUND(AVG(actual_pnl), 2) as avg_pnl,
    ROUND(SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate
FROM ml_predictions
GROUP BY symbol
ORDER BY total_pnl DESC
LIMIT 15`,
    ml_accuracy: `SELECT
    recommendation,
    COUNT(*) as total,
    SUM(hybrid_correct) as correct,
    ROUND(SUM(hybrid_correct) * 100.0 / COUNT(*), 1) as accuracy_pct
FROM ml_predictions
GROUP BY recommendation`,
    monthly_pnl: `SELECT
    strftime('%Y-%m', entry_ts) as month,
    COUNT(*) as trades,
    ROUND(SUM(actual_pnl), 2) as total_pnl,
    ROUND(AVG(actual_pnl), 2) as avg_pnl
FROM ml_predictions
GROUP BY month
ORDER BY month DESC
LIMIT 12`,
    win_rate: `SELECT
    am.account_id,
    am.total_trades,
    ROUND(am.win_rate * 100, 1) as win_rate_pct,
    ROUND(am.net_pnl, 2) as net_pnl,
    ROUND(am.sharpe_ratio, 2) as sharpe
FROM account_metrics am
ORDER BY am.net_pnl DESC`
};

function loadExampleQuery() {
    const select = document.getElementById('queryExamples');
    const textarea = document.getElementById('sqlQuery');

    if (select.value && exampleQueries[select.value]) {
        textarea.value = exampleQueries[select.value];
    }

    select.value = '';
}

function clearQuery() {
    document.getElementById('sqlQuery').value = '';
    document.getElementById('queryResults').innerHTML = `
        <div class="query-results-placeholder">
            <p>Query results will appear here</p>
        </div>
    `;
}

async function executeQuery() {
    const query = document.getElementById('sqlQuery').value.trim();
    const resultsContainer = document.getElementById('queryResults');

    if (!query) {
        resultsContainer.innerHTML = `
            <div class="query-error">
                <p>Please enter a SQL query</p>
            </div>
        `;
        return;
    }

    resultsContainer.innerHTML = '<div class="query-loading">Executing query...</div>';

    try {
        const response = await fetch('/api/database/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `
                <div class="query-error">
                    <strong>Error:</strong> ${data.error}
                </div>
            `;
            return;
        }

        // Display results
        if (data.rows.length === 0) {
            resultsContainer.innerHTML = `
                <div class="query-empty">
                    <p>Query returned no results</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="query-info">
                <span>${data.row_count} row${data.row_count !== 1 ? 's' : ''} returned</span>
                ${data.truncated ? '<span class="truncated-warning">(Results truncated to 1000 rows)</span>' : ''}
            </div>
            <div class="query-table-wrapper">
                <table class="query-results-table">
                    <thead>
                        <tr>
                            ${data.columns.map(c => `<th>${c}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.rows.forEach(row => {
            html += '<tr>';
            data.columns.forEach(col => {
                let val = row[col];
                if (val === null) val = '<span class="null-value">NULL</span>';
                else if (typeof val === 'number') val = formatQueryValue(val);
                html += `<td>${val}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        resultsContainer.innerHTML = html;

    } catch (error) {
        resultsContainer.innerHTML = `
            <div class="query-error">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

function formatQueryValue(val) {
    if (Number.isInteger(val)) return val.toLocaleString();
    if (typeof val === 'number') return val.toFixed(2);
    return val;
}

// =============================================================================
// AI INVESTMENT ADVISOR (Gemini-powered)
// =============================================================================
let advisorHistory = [];
let advisorAvailable = false;

async function checkAdvisorStatus() {
    const statusCard = document.getElementById('advisorStatusCard');

    try {
        const response = await fetch('/api/advisor/status');
        const data = await response.json();

        advisorAvailable = data.available;

        if (data.available) {
            statusCard.innerHTML = `
                <div class="status-indicator online">
                    <span class="status-dot"></span>
                    <span class="status-text">AI Advisor Online - Powered by ${data.model}</span>
                </div>
                <div class="status-features">
                    <h4>Available Features:</h4>
                    <ul>
                        ${data.features.map(f => `<li>${f}</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            statusCard.innerHTML = `
                <div class="status-indicator offline">
                    <span class="status-dot"></span>
                    <span class="status-text">AI Advisor Unavailable</span>
                </div>
                <div class="status-error">
                    <p>The AI Advisor requires the google-genai package. Please install it to use this feature.</p>
                </div>
            `;
        }
    } catch (error) {
        statusCard.innerHTML = `
            <div class="status-indicator offline">
                <span class="status-dot"></span>
                <span class="status-text">AI Advisor Error</span>
            </div>
            <div class="status-error">
                <p>Error connecting to AI Advisor: ${error.message}</p>
            </div>
        `;
    }
}

function sendAdvisorSuggestion(text) {
    document.getElementById('advisor-input').value = text;
    sendAdvisorMessage();
}

function handleAdvisorKeypress(event) {
    if (event.key === 'Enter') {
        sendAdvisorMessage();
    }
}

async function sendAdvisorMessage() {
    const input = document.getElementById('advisor-input');
    const messagesContainer = document.getElementById('advisor-messages');
    const message = input.value.trim();

    if (!message) return;

    // Clear input
    input.value = '';

    // Add user message to chat
    const userMessageEl = document.createElement('div');
    userMessageEl.className = 'advisor-message user';
    userMessageEl.innerHTML = `
        <div class="message-avatar">&#128100;</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    messagesContainer.appendChild(userMessageEl);

    // Add loading indicator
    const loadingEl = document.createElement('div');
    loadingEl.className = 'advisor-message assistant loading';
    loadingEl.innerHTML = `
        <div class="message-avatar">&#129302;</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            <p class="loading-text">Analyzing your question...</p>
        </div>
    `;
    messagesContainer.appendChild(loadingEl);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        const response = await fetch('/api/advisor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: advisorHistory
            })
        });

        const data = await response.json();

        // Remove loading indicator
        loadingEl.remove();

        if (data.error) {
            const errorEl = document.createElement('div');
            errorEl.className = 'advisor-message assistant error';
            errorEl.innerHTML = `
                <div class="message-avatar">&#129302;</div>
                <div class="message-content">
                    <p class="error-text">Error: ${data.error}</p>
                </div>
            `;
            messagesContainer.appendChild(errorEl);
        } else {
            // Update history
            advisorHistory = data.history || [];

            // Add assistant response
            const assistantEl = document.createElement('div');
            assistantEl.className = 'advisor-message assistant';
            assistantEl.innerHTML = `
                <div class="message-avatar">&#129302;</div>
                <div class="message-content">
                    ${formatAdvisorResponse(data.response)}
                </div>
            `;
            messagesContainer.appendChild(assistantEl);
        }
    } catch (error) {
        loadingEl.remove();
        const errorEl = document.createElement('div');
        errorEl.className = 'advisor-message assistant error';
        errorEl.innerHTML = `
            <div class="message-avatar">&#129302;</div>
            <div class="message-content">
                <p class="error-text">Connection error: ${error.message}</p>
            </div>
        `;
        messagesContainer.appendChild(errorEl);
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatAdvisorResponse(text) {
    // Convert markdown-like formatting to HTML
    let html = escapeHtml(text);

    // Convert **bold** to <strong>
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Convert *italic* to <em>
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Convert bullet points
    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Convert numbered lists
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Convert line breaks to paragraphs
    const paragraphs = html.split('\n\n');
    html = paragraphs.map(p => {
        if (p.includes('<li>') || p.includes('<ul>')) return p;
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function runSimulation() {
    const symbol = document.getElementById('sim-symbol').value.trim().toUpperCase();
    const amount = parseFloat(document.getElementById('sim-amount').value);
    const years = parseInt(document.getElementById('sim-years').value);

    if (!symbol) {
        alert('Please enter a stock symbol');
        return;
    }

    if (!amount || amount < 100) {
        alert('Please enter a valid investment amount (minimum $100)');
        return;
    }

    // Construct the message and send to advisor
    const message = `If I invest $${amount.toLocaleString()} in ${symbol} for ${years} year${years > 1 ? 's' : ''}, what returns can I expect?`;
    document.getElementById('advisor-input').value = message;
    sendAdvisorMessage();
}

// Hook into tab switching to check advisor status
const originalSwitchTab = switchTab;
switchTab = function(tabId) {
    originalSwitchTab(tabId);

    if (tabId === 'advisor') {
        checkAdvisorStatus();
    }
};
