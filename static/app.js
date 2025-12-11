// =========================
// Creator Insights Dashboard
// JavaScript for Filters and Charts
// =========================

// Get selected values from checkbox group
function getSelectedValues(containerId, checkboxName) {
    const container = document.getElementById(containerId);
    if (!container) return [];
    const checkboxes = container.querySelectorAll(`input[name="${checkboxName}"]:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

// Select all checkboxes in a filter group
function selectAllFilters(filterType) {
    const container = document.getElementById(`${filterType}-filter`);
    if (container) {
        const checkboxes = container.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = true);
    }
}

// Deselect all checkboxes in a filter group
function deselectAllFilters(filterType) {
    const container = document.getElementById(`${filterType}-filter`);
    if (container) {
        const checkboxes = container.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = false);
    }
}

// Toggle accordion sections
function toggleAccordion(accordionId) {
    const content = document.getElementById(accordionId);
    const header = content.previousElementSibling;
    const accordion = content.parentElement;
    const arrow = header.querySelector('.accordion-arrow');
    
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        arrow.textContent = '▲';
        header.classList.add('active');
        accordion.classList.add('expanded');
    } else {
        content.style.display = 'none';
        arrow.textContent = '▼';
        header.classList.remove('active');
        accordion.classList.remove('expanded');
    }
}

// Apply filters and fetch data
function applyFilters() {
    console.log('=== applyFilters called ===');
    
    // Show loading spinner
    document.getElementById('loading-spinner').style.display = 'flex';
    document.getElementById('charts-container').style.display = 'none';

    // Gather filter values
    const filters = {
        platforms: getSelectedValues('platform-filter', 'platform'),
        post_types: getSelectedValues('post-type-filter', 'post-type'),
        niches: getSelectedValues('niche-filter', 'niche'),
        countries: getSelectedValues('country-filter', 'country'),
        genders: getSelectedValues('gender-filter', 'gender'),
        metrics: getSelectedValues('metric-filter', 'metric'),
        age_range: [
            parseInt(document.getElementById('age-min').value),
            parseInt(document.getElementById('age-max').value)
        ],
        show_all: document.getElementById('show-all').checked
    };

    console.log('Filters:', filters);

    // Fetch data from API
    fetch('/api/get_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(filters)
    })
    .then(response => {
        console.log('Response received:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('=== Data received from API ===');
        console.log('Aggregates:', data.aggregates);
        console.log('Charts keys:', Object.keys(data.charts));
        console.log('Charts:', data.charts);
        console.log('Default graphs:', data.default_graphs);
        
        // Update KPIs
        document.getElementById('total-posts').textContent = data.aggregates.total_posts;
        document.getElementById('avg-engagement').textContent = data.aggregates.avg_engagement_rate;
        document.getElementById('avg-impressions').textContent = data.aggregates.avg_impressions;

        // Update insights
        document.getElementById('insight-hours').textContent = data.insights.best_hours;
        document.getElementById('insight-niche').textContent = data.insights.best_niche;
        document.getElementById('insight-type').textContent = data.insights.best_type;

        // Clear and render charts
        renderCharts(data.charts, data.default_graphs, filters.show_all);

        // Hide loading, show charts
        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('charts-container').style.display = 'grid';
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        const loadingSpinner = document.getElementById('loading-spinner');
        if (loadingSpinner) {
            loadingSpinner.style.display = 'flex';
            loadingSpinner.innerHTML = 
                '<p style="color: #ef4444;">Error loading data. Please try again.</p>';
        }
        document.getElementById('charts-container').style.display = 'none';
    });
}

// Render charts based on data
function renderCharts(charts, defaultGraphs, showAll) {
    console.log('=== renderCharts called ===');
    console.log('showAll:', showAll);
    console.log('defaultGraphs:', defaultGraphs);
    console.log('charts object:', charts);
    
    const container = document.getElementById('charts-container');
    container.innerHTML = '';

    // Define chart order for layout (pairs for 2-column grid)
    const chartOrder = [
        'country_map', 'content_mix',
        'heatmap', 'funnel',
        'viral_patterns', 'platform_radar',
        'best_hour', 'best_day',
        'age', 'gender',
        'virality', 'niche_perf',
        'platform_comp', 'weekend',
        'metrics_heatmap'
    ];

    // Determine which charts to show
    let chartsToShow = [];
    
    if (showAll) {
        // Show all available charts
        chartsToShow = chartOrder.filter(key => {
            const hasChart = charts[key] && Object.keys(charts[key]).length > 0;
            console.log(`Chart ${key}: exists=${!!charts[key]}, hasKeys=${hasChart}`);
            return hasChart;
        });
        console.log('Charts to show (all):', chartsToShow);
    } else {
        // Show only default graphs for user type
        const graphMapping = {
            'heatmap': 'heatmap',
            'content_mix': 'content_mix',
            'viral_patterns': 'viral_patterns',
            'funnel': 'funnel',
            'platform_radar': 'platform_radar',
            'weekend': 'weekend',
            'best_hour': 'best_hour',
            'best_day': 'best_day',
            'age_gender': ['age', 'gender'],
            'age_dist': 'age',
            'virality': 'virality',
            'niche_perf': 'niche_perf',
            'platform_comp': 'platform_comp',
            'country_map': 'country_map'
        };

        defaultGraphs.forEach(graph => {
            const mapped = graphMapping[graph];
            if (Array.isArray(mapped)) {
                chartsToShow.push(...mapped);
            } else if (mapped) {
                chartsToShow.push(mapped);
            }
        });

        // Remove duplicates and maintain order
        const seen = new Set();
        chartsToShow = chartOrder.filter(key => {
            if (chartsToShow.includes(key) && !seen.has(key) && charts[key] && Object.keys(charts[key]).length > 0) {
                seen.add(key);
                return true;
            }
            return false;
        });
        console.log('Charts to show (default):', chartsToShow);
    }

    // Create chart containers and render
    chartsToShow.forEach(chartKey => {
        console.log(`--- Rendering chart: ${chartKey} ---`);
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.id = `chart-${chartKey}`;
        container.appendChild(chartDiv);

        try {
            const chartData = charts[chartKey];
            console.log(`${chartKey} data:`, chartData);
            console.log(`${chartKey} has data property:`, !!chartData.data);
            console.log(`${chartKey} has layout property:`, !!chartData.layout);
            
            Plotly.newPlot(`chart-${chartKey}`, chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
            });
            console.log(`${chartKey} rendered successfully`);
        } catch (error) {
            console.error(`Error rendering chart ${chartKey}:`, error);
            chartDiv.innerHTML = `<p style="color: #6b7280; text-align: center; padding: 20px;">Unable to render chart: ${error.message}</p>`;
        }
    });

    // If no charts to show
    if (chartsToShow.length === 0) {
        console.log('No charts to display');
        container.innerHTML = '<p style="color: #6b7280; text-align: center; padding: 40px;">No data available for selected filters</p>';
    } else {
        console.log(`Total charts rendered: ${chartsToShow.length}`);
    }
}

// Enable "Enter" key to apply filters
document.addEventListener('DOMContentLoaded', function() {
    const filterInputs = document.querySelectorAll('.sidebar select, .sidebar input');
    filterInputs.forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                applyFilters();
            }
        });
    });
});
