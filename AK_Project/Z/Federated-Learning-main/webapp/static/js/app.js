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

function getCookieToken() {
    const cookies = document.cookie.split(';');
    for (const c of cookies) {
        const [name, value] = c.trim().split('=');
        if (name === 'access_token') {
            return decodeURIComponent(value).replace('Bearer ', '');
        }
    }
    return '';
}

async function stopProc(processId, role) {
    role = role || (window.location.pathname.startsWith('/admin') ? 'admin' : 'hospital');
    const prefix = role === 'admin' ? '/api/admin' : '/api/hospital';
    const ok = confirm(`Stop process ${processId}?`);
    if (!ok) return;
    try {
        const r = await apiPost(`${prefix}/processes/${processId}/stop`);
        showToast(r.message || 'Stop signal sent', r.success ? 'info' : 'warning');
        setTimeout(() => window.dispatchEvent(new Event('proc-refresh')), 600);
    } catch(e) { showToast('Stop failed: ' + String(e), 'error'); }
}

function openPathPicker(role, onSelect, opts) {
    role = role || (window.location.pathname.startsWith('/admin') ? 'admin' : 'hospital');
    const prefix = role === 'admin' ? '/api/admin' : '/api/hospital';
    const mask = (opts && opts.ext) ? new RegExp(`\\.(${opts.ext.join('|')})$`, 'i') : null;
    const modal = document.createElement('div');
    modal.setAttribute('role','dialog');
    modal.style.cssText = 'position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:9999;display:flex;align-items:center;justify-content:center;padding:20px';
    modal.innerHTML = `
        <div style="background:#fff;border-radius:14px;width:min(900px,100%);min-height:460px;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(15,23,42,.3)">
            <div style="padding:14px 18px;border-bottom:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:center">
                <h3 style="margin:0;font-size:16px">Pick a file or folder from your workspace</h3>
                <button id="pp_close" style="background:#f1f5f9;border:none;padding:6px 10px;border-radius:6px;cursor:pointer">Close</button>
            </div>
            <div style="display:grid;grid-template-columns:200px 1fr;flex:1;min-height:0">
                <div id="pp_roots" style="padding:10px;border-right:1px solid #f1f5f9;overflow:auto"></div>
                <div style="display:flex;flex-direction:column;min-height:0">
                    <div id="pp_crumbs" style="padding:8px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#64748b"></div>
                    <div id="pp_files" style="flex:1;overflow:auto;padding:4px 0"></div>
                </div>
            </div>
            <div style="padding:12px 18px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:center">
                <div id="pp_sel" style="font-size:12px;color:#64748b">No selection</div>
                <button id="pp_ok" disabled style="background:#2563eb;color:#fff;border:none;padding:8px 16px;border-radius:8px;font-weight:600;cursor:pointer">Select</button>
            </div>
        </div>`;
    document.body.appendChild(modal);
    let curRoot=null, curPath='', curFile=null;
    const el = (id) => modal.querySelector('#'+id);
    const loadRoots = async () => {
        const r = await apiGet(`${prefix}/workspace/roots`);
        el('pp_roots').innerHTML = (r.roots||[]).map(rt => `
            <div data-root="${rt.key}" class="pp_root" style="padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:4px;font-size:13px;color:#334155">
                ${rt.writable?'📂':'📁'} ${rt.label||rt.key} <small style="color:#94a3b8">· ${rt.file_count||0}</small>
            </div>
        `).join('');
        modal.querySelectorAll('.pp_root').forEach(d => d.addEventListener('click', () => {
            curRoot = d.dataset.root; curPath = ''; curFile=null;
            modal.querySelectorAll('.pp_root').forEach(x => x.style.background = x === d ? '#eff6ff' : '');
            modal.querySelectorAll('.pp_root').forEach(x => x.style.color = x === d ? '#1e40af' : '#334155');
            loadFiles();
        }));
    };
    const fmtSize = b => b===undefined?'':b<1024?b+' B':b<1024*1024?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(2)+' MB';
    const loadFiles = async () => {
        if (!curRoot) return;
        const parts = curPath.split('/').filter(Boolean);
        el('pp_crumbs').innerHTML = `<span style="color:#2563eb;cursor:pointer" data-r="">${curRoot}</span>` +
            parts.map((_,i) => `<span style="color:#cbd5e1;margin:0 2px">/</span><span style="color:#2563eb;cursor:pointer" data-p="${parts.slice(0,i+1).join('/')}">${parts[i]}</span>`).join('');
        el('pp_crumbs').querySelectorAll('span[data-r]').forEach(s => s.onclick = () => { curPath=''; loadFiles(); });
        el('pp_crumbs').querySelectorAll('span[data-p]').forEach(s => s.onclick = () => { curPath = s.dataset.p; loadFiles(); });
        const r = await apiGet(`${prefix}/workspace/ls?root=${encodeURIComponent(curRoot)}&path=${encodeURIComponent(curPath)}`);
        const ent = (r.entries||[]).sort((a,b)=>{ if(a.is_dir!==b.is_dir) return a.is_dir?-1:1; return a.name.localeCompare(b.name); });
        const up = curPath ? `<div class="pp_row up" style="padding:6px 14px;color:#64748b;cursor:pointer;font-style:italic;font-size:13px">⬆ .. parent</div>` : '';
        el('pp_files').innerHTML = up + ent.map(e => {
            const ok = mask ? (e.is_dir || mask.test(e.name)) : true;
            return `<div class="pp_row" style="padding:6px 14px;cursor:pointer;font-size:13px;display:grid;grid-template-columns:1fr 100px;gap:8px;align-items:center;color:${ok?'#0f172a':'#cbd5e1'}" data-name="${e.name}" data-dir="${e.is_dir?'1':'0'}" data-dl="${e.downloadable?'1':'0'}">
                <div>${e.is_dir?'📁 ':'📄 '}${e.name}</div><div style="color:#94a3b8;font-size:11px">${fmtSize(e.size_bytes)}</div>
            </div>`;
        }).join('');
        el('pp_files').querySelectorAll('.pp_row.up').forEach(d => d.addEventListener('click', () => {
            curPath = curPath.split('/').slice(0,-1).join('/'); loadFiles();
        }));
        el('pp_files').querySelectorAll('.pp_row[data-dir]').forEach(d => d.addEventListener('click', () => {
            if (d.dataset.dir === '1') {
                curPath = curPath ? (curPath+'/'+d.dataset.name) : d.dataset.name;
                curFile = null;
                updateSel();
                loadFiles();
            } else if (!mask || mask.test(d.dataset.name)) {
                curFile = { name: d.dataset.name, path: (curPath?curPath+'/':'') + d.dataset.name, root: curRoot, downloadable: d.dataset.dl==='1'};
                el('pp_files').querySelectorAll('.pp_row[data-dir]').forEach(x => x.style.background = x===d ? '#eff6ff' : '');
                updateSel();
            }
        }));
        updateSel();
    };
    const updateSel = () => {
        el('pp_sel').textContent = curFile ? `Selected: ${curFile.root}/${curFile.path}` : (curRoot ? `Browsing ${curRoot}/${curPath}` : 'No selection');
        el('pp_ok').disabled = !curFile;
    };
    el('pp_close').onclick = () => document.body.removeChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) document.body.removeChild(modal); });
    el('pp_ok').onclick = () => { if (curFile) { onSelect(curFile); document.body.removeChild(modal); } };
    loadRoots();
}

function handleFileUploads(formId, endpoint, onDone, extraFields) {
    const form = typeof formId === 'string' ? document.getElementById(formId) : formId;
    if (!form) return;
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const token = getCookieToken();
        const inputs = form.querySelectorAll('input[type=file]');
        const fd = new FormData();
        let hasAny = false;
        inputs.forEach(inp => {
            for (const f of (inp.files||[])) { fd.append(inp.name || 'files', f); hasAny = true; }
        });
        if (!hasAny) { showToast('Select at least one file','warning'); return; }
        if (extraFields) Object.entries(extraFields).forEach(([k,v]) => fd.append(k, v));
        const extras = form.querySelectorAll('input[data-upload]');
        extras.forEach(x => { if (x.value) fd.append(x.dataset.upload, x.value); });
        try {
            const res = await fetch(endpoint, { method:'POST', headers: { 'Authorization': 'Bearer ' + token }, body: fd });
            const data = await res.json();
            if (typeof onDone === 'function') onDone(data);
            else if (data.success) showToast(`Uploaded ${data.saved.length} file(s)`, 'success');
            else showToast((data.errors||[])[0] || 'Upload failed', 'error');
        } catch(err) { showToast('Upload failed: ' + String(err), 'error'); }
    });
}

function renderCSVView(elementId, columns, rows, opts) {
    const host = document.getElementById(elementId);
    if (!host) return;
    const limit = (opts && opts.limit) || 500;
    const truncRows = (rows||[]).slice(0, limit);
    const wrap = document.createElement('div');
    wrap.style.cssText = 'overflow:auto;border:1px solid #e2e8f0;border-radius:8px;max-height:100%';
    const t = document.createElement('table');
    t.className = 'csv-view-table';
    t.style.cssText = 'border-collapse:collapse;width:100%;font-size:12px';
    t.innerHTML = `<thead><tr style="background:#f8fafc">${(columns||[]).map(c=>`<th style="padding:8px 10px;text-align:left;border-bottom:2px solid #e2e8f0;color:#0f172a;white-space:nowrap;position:sticky;top:0">${c}</th>`).join('')}</tr></thead>
        <tbody>${truncRows.map(r=>`<tr style="border-bottom:1px solid #f8fafc">${r.map(v=>`<td style="padding:6px 10px;color:#334155;white-space:nowrap">${v===null||v===undefined?'':String(v)}</td>`).join('')}</tr>`).join('')}</tbody>`;
    wrap.appendChild(t);
    host.innerHTML = '';
    host.appendChild(wrap);
    if (rows && rows.length > limit) {
        const note = document.createElement('div');
        note.style.cssText = 'padding:6px 8px;font-size:11px;color:#94a3b8';
        note.textContent = `Showing ${limit} of ${rows.length} rows`;
        host.appendChild(note);
    }
}

function renderTextView(elementId, content, kind) {
    const host = document.getElementById(elementId);
    if (!host) return;
    const pre = document.createElement('pre');
    pre.className = 'text-view-block';
    pre.style.cssText = 'background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px;font-family:Consolas,monospace;font-size:12px;line-height:1.5;overflow:auto;white-space:pre-wrap;margin:0;max-height:100%';
    pre.textContent = content || '';
    host.innerHTML = '';
    host.appendChild(pre);
}

function renderImageView(elementId, src, alt) {
    const host = document.getElementById(elementId);
    if (!host) return;
    host.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;justify-content:center;padding:12px;max-height:100%';
    const img = document.createElement('img');
    img.src = src;
    img.alt = alt || 'preview';
    img.style.cssText = 'max-width:100%;max-height:100%;border-radius:8px;box-shadow:0 6px 18px rgba(15,23,42,.12)';
    wrap.appendChild(img);
    host.appendChild(wrap);
}

function subscribeProcessWS(processId, onLog, onStatus) {
    const token = getCookieToken();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/process/${processId}?token=${encodeURIComponent('Bearer ' + token)}`;
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
        try {
            const m = JSON.parse(ev.data);
            if (m.type === 'process_log' && typeof onLog === 'function') onLog(m);
            if (m.type === 'process_status' && typeof onStatus === 'function') onStatus(m);
        } catch(e) {}
    };
    ws.onopen = () => { if (typeof onStatus === 'function') onStatus({ type:'process_status', status:'connected' }); };
    return ws;
}
