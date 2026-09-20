function getAuthHeaders() {
    const cookies = document.cookie.split(';');
    let token = null;
    for (const c of cookies) {
        const [name, value] = c.trim().split('=');
        if (name === 'access_token') {
            token = decodeURIComponent(value).replace('Bearer ', '');
            break;
        }
    }
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

async function apiGet(url) {
    const res = await fetch(url, { headers: getAuthHeaders() });
    if (res.status === 401) {
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }
    return res.json();
}

async function apiPost(url, body = {}) {
    const res = await fetch(url, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(body)
    });
    if (res.status === 401) {
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }
    return res.json();
}

async function apiPatch(url, body = {}) {
    const res = await fetch(url, {
        method: 'PATCH',
        headers: getAuthHeaders(),
        body: JSON.stringify(body)
    });
    if (res.status === 401) {
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }
    return res.json();
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) {
        const el = document.createElement('div');
        el.id = 'toastContainer';
        el.className = 'toast-container';
        document.body.appendChild(el);
    }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    document.getElementById('toastContainer').appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function formatTime(date) {
    const d = new Date(date);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDateTime(date) {
    if (!date) return '-';
    const d = new Date(date);
    return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function getInitials(name) {
    if (!name) return 'U';
    const parts = name.split(/[_\s]/);
    if (parts.length === 1) return parts[0].charAt(0).toUpperCase();
    return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

function updateCurrentDateTime() {
    const el = document.getElementById('currentDateTime');
    if (el) {
        const now = new Date();
        const options = { month: 'short', day: 'numeric', year: 'numeric' };
        el.textContent = now.toLocaleDateString('en-US', options);
    }
    const elTime = document.getElementById('currentTime');
    if (elTime) {
        const now = new Date();
        elTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
}

setInterval(updateCurrentDateTime, 30000);

async function logout() {
    try {
        await apiPost('/api/auth/logout');
    } catch (e) {}
    document.cookie = 'access_token=; path=/; max-age=0';
    window.location.href = '/login';
}

function drawLineChart(canvasId, datasets, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: { boxWidth: 12, padding: 16, font: { size: 12 } }
            }
        },
        scales: {
            x: { grid: { display: false }, ticks: { font: { size: 11 } } },
            y: {
                grid: { color: '#f1f5f9' },
                ticks: { font: { size: 11 } },
                beginAtZero: false
            }
        },
        interaction: { intersect: false, mode: 'index' }
    };

    return new Chart(canvas, {
        type: 'line',
        data: {
            labels: datasets.labels,
            datasets: datasets.series.map((s, i) => ({
                label: s.label,
                data: s.data,
                borderColor: s.color || ['#2563eb','#10b981','#f59e0b','#8b5cf6'][i % 4],
                backgroundColor: (s.color || ['#2563eb','#10b981','#f59e0b','#8b5cf6'][i % 4]) + '20',
                tension: 0.35,
                borderWidth: 2.5,
                pointRadius: 3,
                pointHoverRadius: 5,
                fill: !!s.fill
            }))
        },
        options: { ...defaultOptions, ...options }
    });
}

function drawBarChart(canvasId, datasets, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels: datasets.labels,
            datasets: datasets.series.map((s, i) => ({
                label: s.label,
                data: s.data,
                backgroundColor: s.color || ['#2563eb','#10b981','#f59e0b','#8b5cf6'][i % 4],
                borderRadius: 6,
                maxBarThickness: 40
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 12, font: { size: 12 } } } },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: { grid: { color: '#f1f5f9' }, ticks: { font: { size: 11 } } }
            }
        }
    });
}

function drawDoughnutChart(canvasId, labels, data, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    return new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { display: false }
            }
        }
    });
}
