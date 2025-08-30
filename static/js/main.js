// FILE: static/js/main.js

document.addEventListener('DOMContentLoaded', () => {

    const POLLING_INTERVAL = 5000;
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    // [Design System] Plotly layout for a refined, elegant academic style
    const commonLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            family: 'Inter, sans-serif',
            size: 13,
            color: '#212529'
        },
        xaxis: {
            gridcolor: '#dee2e6',
            linecolor: '#adb5bd',
            title: {
                text: 'Iteration',
                font: { size: 14, color: '#6c757d' },
                standoff: 20
            },
            zeroline: false,
            tickfont: { color: '#6c757d' }
        },
        yaxis: {
            gridcolor: '#dee2e6',
            linecolor: '#adb5bd',
            title: {
                font: { size: 14, color: '#6c757d' },
                standoff: 20
            },
            zeroline: false,
            tickfont: { color: '#6c757d' }
        },
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
            font: { size: 13 }
        },
        margin: { l: 70, r: 30, b: 60, t: 30 },
        hovermode: 'x unified',
        hoverlabel: {
            bgcolor: "#ffffff",
            bordercolor: "#dee2e6",
            font: {
                family: 'Inter, sans-serif',
                size: 13,
                color: "#212529"
            },
            align: 'left',
            namelength: -1
        }
    };

    const commonConfig = {
        responsive: true,
        displayModeBar: false // Keep Plotly toolbar hidden for a cleaner look
    };

    const mergeDeep = (target, ...sources) => {
        const isObject = (obj) => obj && typeof obj === 'object';
        const output = { ...target };
        sources.forEach(source => {
            if (isObject(source)) {
                Object.keys(source).forEach(key => {
                    if (isObject(source[key])) {
                        if (!(key in output))
                            Object.assign(output, { [key]: source[key] });
                        else
                            output[key] = mergeDeep(output[key], source[key]);
                    } else {
                        Object.assign(output, { [key]: source[key] });
                    }
                });
            }
        });
        return output;
    };

    // [Design System] Professional color palette
    const colors = {
        primary: '#0d6efd',     // Accent Blue
        value: '#198754',       // Value/Best Model (Success Green)
        policy: '#fd7e14',      // Policy (Vibrant Orange)
        generalization: '#6f42c1', // Generalization (Deep Purple)
        heft: '#6c757d',        // Baseline (Secondary Grey)
        promotion: '#dc3545',   // Promotion (Danger Red)
        lr: '#0dcaf0',          // Learning Rate (Info Cyan)
    };

    // --- Chart Drawing Functions ---
    const plotCharts = (data) => {
        if (!data || Object.keys(data).length === 0 || !data.iteration || data.iteration.length === 0) return;

        // 1. Training Loss Chart
        const lossTraces = [
            { x: data.iteration, y: data.avg_total_loss, mode: 'lines+markers', name: 'Total Loss', line: { color: colors.primary, width: 2.5 }, marker: { size: 4 } },
            { x: data.iteration, y: data.avg_value_loss, mode: 'lines+markers', name: 'Value Loss', line: { color: colors.value, width: 1.5, dash: 'dash' }, marker: { size: 4, symbol: 'x' } },
            { x: data.iteration, y: data.avg_policy_loss, mode: 'lines+markers', name: 'Policy Loss', line: { color: colors.policy, width: 1.5, dash: 'dash' }, marker: { size: 4, symbol: 'cross' } }
        ];
        Plotly.react('loss_chart', lossTraces, mergeDeep(commonLayout, { yaxis: { title: { text: 'Loss' } } }), commonConfig);

        // 2. Learning Rate & Guidance Epsilon Chart
        const lrGuidanceTraces = [
            { x: data.iteration, y: data.learning_rate, name: 'Learning Rate', type: 'scatter', mode: 'lines', yaxis: 'y1', line: { color: colors.lr } },
            { x: data.iteration, y: data.guidance_epsilon, name: 'Guidance ε', type: 'scatter', mode: 'lines', yaxis: 'y2', line: { color: colors.policy, dash: 'dot' } }
        ];
        const lrLayout = mergeDeep(commonLayout, {
            yaxis: { title: 'Learning Rate', side: 'left' },
            yaxis2: { title: 'Guidance Epsilon', overlaying: 'y', side: 'right', showgrid: false, range: [0, Math.max(...data.guidance_epsilon.filter(v => v !== null), 0.5)] },
             legend: { y: 1.15 }
        });
        Plotly.react('lr_guidance_chart', lrGuidanceTraces, lrLayout, commonConfig);

        // 3. Evaluation Arena Chart
        const evaluationTraces = [
            { x: data.iteration, y: data.avg_cand_makespan, mode: 'lines+markers', name: 'Candidate Model', line: { color: colors.primary }, marker: { symbol: 'circle', size: 6 } },
            { x: data.iteration, y: data.avg_best_makespan, mode: 'lines+markers', name: 'Best Model', line: { color: colors.value }, marker: { symbol: 'diamond', size: 6 } },
            { x: data.iteration, y: data.avg_heft_makespan, mode: 'lines+markers', name: 'HEFT Baseline', line: { color: colors.heft, dash: 'dot', width: 2 }, marker: { symbol: 'circle-open', size: 5 } }
        ];
        const promotedIterations = data.iteration.filter((_, i) => data.promoted && data.promoted[i] == 1);
        if (promotedIterations.length > 0) {
            const promotedMakespans = data.avg_cand_makespan.filter((_, i) => data.promoted && data.promoted[i] == 1);
            evaluationTraces.push({
                x: promotedIterations,
                y: promotedMakespans,
                mode: 'markers',
                name: 'Model Promoted',
                marker: { symbol: 'star', color: colors.promotion, size: 14, line: { color: '#ffffff', width: 1.5 } },
                hoverinfo: 'text',
                text: promotedIterations.map(iter => `Promoted at Iteration ${iter}`)
            });
        }
        Plotly.react('evaluation_chart', evaluationTraces, mergeDeep(commonLayout, { yaxis: { title: { text: 'Avg. Makespan' } } }), commonConfig);

        // 4. Improvement vs HEFT Chart
        const improvementTraces = [
            { x: data.iteration, y: data.improvement_vs_heft, name: 'Improvement %', type: 'scatter', mode: 'lines+markers', line: { color: colors.primary, shape: 'spline' }, marker: {size: 5} }
        ];
        Plotly.react('improvement_chart', improvementTraces, mergeDeep(commonLayout, { yaxis: { title: { text: 'Improvement (%)' }, zeroline: true, zerolinewidth: 2, zerolinecolor: colors.heft } }), commonConfig);

        // 5. Self-Play Raw Reward Distribution Chart
        const rewardTraces = [
            { x: data.iteration, y: data.reward_mean, name: 'Mean Reward', type: 'scatter', mode: 'lines', line: { color: colors.generalization, width: 2.5 } },
            {
                x: [...data.iteration, ...[...data.iteration].reverse()],
                y: [...(data.reward_mean.map((m, i) => m + (data.reward_std_dev[i] || 0))), ...[...(data.reward_mean.map((m, i) => m - (data.reward_std_dev[i] || 0)))].reverse()],
                fill: 'toself',
                fillcolor: 'rgba(111, 66, 193, 0.2)',
                line: { color: 'transparent' },
                hoverinfo: 'none',
                name: 'Std Dev Range',
                showlegend: true
            }
        ];
        Plotly.react('reward_chart', rewardTraces, mergeDeep(commonLayout, { yaxis: { title: { text: 'Raw Reward (Neg. Makespan)' } } }), commonConfig);
    };

    // [Optimized] Evaluation Makespan Distribution Chart (Grouped Box Plot)
    const plotEvalDetailsChart = (data) => {
        if (!data || !data.iterations || data.iterations.length === 0) {
            Plotly.purge('evaluation_details_chart');
            return;
        }

        const traces = [
            { type: 'box', name: 'Candidate Model', marker: { color: colors.primary } },
            { type: 'box', name: 'Best Model', marker: { color: colors.value } },
            { type: 'box', name: 'HEFT Baseline', marker: { color: colors.heft } }
        ];

        // This data transformation is key for grouped box plots
        traces[0].x = data.iterations;
        traces[0].y = data.candidate_makespans.flat();
        traces[0].boxpoints = 'Outliers';

        traces[1].x = data.iterations;
        traces[1].y = data.best_model_makespans.flat();
        traces[1].boxpoints = 'Outliers';

        traces[2].x = data.iterations;
        traces[2].y = data.heft_makespans.flat();
        traces[2].boxpoints = 'Outliers';

        // Custom transformation to create correct groups for plotly
        const createGroupedX = (iterations, groupName) => {
            let result = [];
            iterations.forEach(iter => {
                // This creates a structure like [['Iter 1', 'Cand'], ['Iter 1', 'Cand'], ...]
                // Plotly uses this multi-level array for grouping on a categorical axis
                result.push(Array(data.candidate_makespans[iterations.indexOf(iter)].length).fill([iter, groupName]));
            });
            return result.flat();
        };

        const layout = mergeDeep(commonLayout, {
            yaxis: { title: { text: 'Makespan Distribution' } },
            xaxis: { type: 'category' },
            boxmode: 'group',
            legend: { y: 1.15 }
        });

        Plotly.react('evaluation_details_chart', traces, layout, commonConfig);
    };


    // --- Data Fetching & Dashboard Update Logic ---
    let lastDataHash = '';
    const updateDashboard = async () => {
        try {
            const [mainResponse, evalDetailsResponse] = await Promise.all([
                fetch('/data'),
                fetch('/eval_details_data')
            ]);

            if (!mainResponse.ok) throw new Error(`/data fetch failed: ${mainResponse.statusText}`);
            const mainDataText = await mainResponse.text();

            if (mainDataText === lastDataHash) { // No changes, skip re-rendering
                 statusIndicator.className = 'status-indicator live';
                 statusText.textContent = `Live (No Change) | ${new Date().toLocaleTimeString()}`;
                 return;
            }
            lastDataHash = mainDataText;

            const mainData = JSON.parse(mainDataText.replace(/NaN/g, 'null'));
            if (mainData.error) throw new Error(mainData.error);
            plotCharts(mainData);

            if (!evalDetailsResponse.ok) throw new Error(`/eval_details_data failed: ${evalDetailsResponse.statusText}`);
            const evalDetailsData = await evalDetailsResponse.json();
            if (evalDetailsData.error) throw new Error(evalDetailsData.error);
            plotEvalDetailsChart(evalDetailsData);

            statusIndicator.className = 'status-indicator live';
            statusText.textContent = `Live | Last update: ${new Date().toLocaleTimeString()}`;
        } catch (error) {
            console.error("Failed to fetch or plot data:", error);
            statusIndicator.className = 'status-indicator error';
            statusText.textContent = `Error: ${error.message}`;
        }
    };

    // --- Fullscreen & Interactivity ---
    const modal = document.getElementById('fullscreen-modal');
    const fullscreenContainer = document.getElementById('fullscreen-chart-container');
    const closeBtn = document.querySelector('.close-btn');

    document.querySelectorAll('.chart-container').forEach(container => {
        container.addEventListener('click', () => {
            const chartId = container.dataset.chartId;
            const sourceChartNode = document.getElementById(chartId);

            if (sourceChartNode && sourceChartNode.data) {
                const newLayout = JSON.parse(JSON.stringify(sourceChartNode.layout));
                newLayout.autosize = true; // Ensure it fills the container
                Plotly.react(fullscreenContainer, sourceChartNode.data, newLayout, commonConfig);
                modal.style.display = 'flex';
                // Trigger resize after modal is visible
                setTimeout(() => Plotly.Plots.resize(fullscreenContainer), 50);
            }
        });
    });

    const closeModal = () => {
        if (modal.style.display === 'flex') {
            modal.style.display = 'none';
            Plotly.purge(fullscreenContainer);
        }
    };
    closeBtn.addEventListener('click', (e) => { e.stopPropagation(); closeModal(); });
    modal.addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => { if (e.key === "Escape") closeModal(); });

    // --- Initialization & Polling ---
    updateDashboard();
    setInterval(updateDashboard, POLLING_INTERVAL);
});