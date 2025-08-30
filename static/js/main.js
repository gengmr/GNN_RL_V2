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

    // ============================ [ 代码修改 1/3 - 新增 ] ============================
    // [原因] 创建一个新的绘图函数来可视化预训练阶段的评估结果。
    // [方案] 1. 函数接收从新API获取的数据。
    //        2. 检查数据是否有效，如果无效则清空图表。
    //        3. 创建两个trace：一个用于模型makespan，一个用于HEFT makespan。
    //        4. 使用不同的颜色和线型以区分。
    //        5. 自定义布局，将X轴标题设为 'Epoch'。
    //        6. 使用Plotly.react绘制图表。
    const plotPretrainEvalChart = (data) => {
        if (!data || !data.epoch || data.epoch.length === 0) {
            Plotly.purge('pretrain_eval_chart'); // 清空图表以防显示旧数据
            return;
        }

        const pretrainTraces = [
            { x: data.epoch, y: data.model_makespan, mode: 'lines+markers', name: 'Pre-trained Model', line: { color: colors.primary, width: 2.5 }, marker: { size: 5 } },
            { x: data.epoch, y: data.heft_makespan, mode: 'lines', name: 'HEFT Baseline', line: { color: colors.heft, width: 2, dash: 'dash' } }
        ];

        const layout = mergeDeep(commonLayout, {
             xaxis: { title: { text: 'Pre-training Epoch' } },
             yaxis: { title: { text: 'Avg. Makespan on Test Set' } }
        });

        Plotly.react('pretrain_eval_chart', pretrainTraces, layout, commonConfig);
    };
    // ========================= [ 修改结束 ] =========================

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

        // 5. Self-Play Value Target Distribution Chart
        const valueTargetTraces = [
            { x: data.iteration, y: data.value_target_mean, name: 'Mean Value Target', type: 'scatter', mode: 'lines', line: { color: colors.generalization, width: 2.5 } },
            {
                x: [...data.iteration, ...[...data.iteration].reverse()],
                y: [...(data.value_target_mean.map((m, i) => m + (data.value_target_std[i] || 0))), ...[...(data.value_target_mean.map((m, i) => m - (data.value_target_std[i] || 0)))].reverse()],
                fill: 'toself',
                fillcolor: 'rgba(111, 66, 193, 0.2)',
                line: { color: 'transparent' },
                hoverinfo: 'none',
                name: 'Std Dev Range',
                showlegend: true
            }
        ];
        Plotly.react('value_target_chart', valueTargetTraces, mergeDeep(commonLayout, { yaxis: { title: { text: 'Value Target (Pre-Normalization)' } } }), commonConfig);
    };

    // [Optimized] Evaluation Makespan Distribution Chart (Grouped Box Plot)
    const plotEvalDetailsChart = (data) => {
        if (!data || !data.iterations || data.iterations.length === 0) {
            Plotly.purge('evaluation_details_chart');
            return;
        }

        const traces = [
            {
                name: 'Candidate Model',
                y: data.candidate_makespans.flat(),
                x: data.iterations.map((iter, i) => Array(data.candidate_makespans[i].length).fill(iter)).flat(),
                type: 'box',
                boxpoints: 'Outliers',
                marker: { color: colors.primary }
            },
            {
                name: 'Best Model',
                y: data.best_model_makespans.flat(),
                x: data.iterations.map((iter, i) => Array(data.best_model_makespans[i].length).fill(iter)).flat(),
                type: 'box',
                boxpoints: 'Outliers',
                marker: { color: colors.value }
            },
            {
                name: 'HEFT Baseline',
                y: data.heft_makespans.flat(),
                x: data.iterations.map((iter, i) => Array(data.heft_makespans[i].length).fill(iter)).flat(),
                type: 'box',
                boxpoints: 'Outliers',
                marker: { color: colors.heft }
            }
        ];

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
            // ============================ [ 代码修改 2/3 - 修改 ] ============================
            // [原因] 需要同时获取主训练数据、评估详情数据和新的预训练评估数据。
            // [方案] 将新的 /pretrain_eval_data 端点添加到 Promise.all 中。
            const [mainResponse, evalDetailsResponse, pretrainEvalResponse] = await Promise.all([
                fetch('/data'),
                fetch('/eval_details_data'),
                fetch('/pretrain_eval_data')
            ]);
            // ========================= [ 修改结束 ] =========================

            if (!mainResponse.ok) throw new Error(`/data fetch failed: ${mainResponse.statusText}`);
            const mainDataText = await mainResponse.text();

            // [优化] 检查所有数据源的哈希，以减少不必要的重绘
            const pretrainDataText = await pretrainEvalResponse.text();
            const combinedHash = mainDataText + pretrainDataText;

            if (combinedHash === lastDataHash) { // No changes, skip re-rendering
                 statusIndicator.className = 'status-indicator live';
                 statusText.textContent = `Live (No Change) | ${new Date().toLocaleTimeString()}`;
                 return;
            }
            lastDataHash = combinedHash;

            const mainData = JSON.parse(mainDataText.replace(/NaN/g, 'null'));
            if (mainData.error) throw new Error(mainData.error);
            plotCharts(mainData);

            if (!evalDetailsResponse.ok) throw new Error(`/eval_details_data failed: ${evalDetailsResponse.statusText}`);
            const evalDetailsData = await evalDetailsResponse.json();
            if (evalDetailsData.error) throw new Error(evalDetailsData.error);
            plotEvalDetailsChart(evalDetailsData);

            // ============================ [ 代码修改 3/3 - 新增 ] ============================
            // [原因] 调用新的绘图函数来渲染预训练评估图表。
            // [方案] 1. 检查预训练评估响应是否成功。
            //        2. 解析JSON数据。
            //        3. 调用 plotPretrainEvalChart 函数。
            if (!pretrainEvalResponse.ok) throw new Error(`/pretrain_eval_data failed: ${pretrainEvalResponse.statusText}`);
            // 使用之前已读取的 pretrainDataText 避免重复请求
            const pretrainEvalData = JSON.parse(pretrainDataText.replace(/NaN/g, 'null'));
            if (pretrainEvalData.error) throw new Error(pretrainEvalData.error);
            plotPretrainEvalChart(pretrainEvalData);
            // ========================= [ 修改结束 ] =========================

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