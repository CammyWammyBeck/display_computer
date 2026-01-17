/**
 * Display Computer Control Panel
 */

class DisplayController {
    constructor() {
        this.ws = null;
        this.state = null;
        this.stats = null;
        this.reconnectInterval = null;
        this.controlsRendered = false;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.connect();
    }

    bindElements() {
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            projectSelector: document.getElementById('project-selector'),
            projectList: document.getElementById('project-list'),
            currentProject: document.getElementById('current-project'),
            projectName: document.getElementById('project-name'),
            projectMode: document.getElementById('project-mode'),
            statsPanel: document.getElementById('stats-panel'),
            controlsPanel: document.getElementById('controls-panel'),
            returnMenu: document.getElementById('return-menu'),
            btnMenu: document.getElementById('btn-menu'),
            btnShutdown: document.getElementById('btn-shutdown'),
        };
    }

    bindEvents() {
        this.elements.returnMenu.addEventListener('click', () => {
            this.sendCommand('return_to_menu');
        });

        this.elements.btnMenu.addEventListener('click', () => {
            this.sendCommand('return_to_menu');
        });

        this.elements.btnShutdown.addEventListener('click', () => {
            if (confirm('Are you sure you want to shutdown?')) {
                this.sendCommand('shutdown');
            }
        });
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log('Connecting to', wsUrl);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('Connected');
            this.setConnectionStatus(true);
            this.clearReconnect();
        };

        this.ws.onclose = () => {
            console.log('Disconnected');
            this.setConnectionStatus(false);
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
    }

    scheduleReconnect() {
        if (this.reconnectInterval) return;
        this.reconnectInterval = setInterval(() => {
            console.log('Attempting to reconnect...');
            this.connect();
        }, 3000);
    }

    clearReconnect() {
        if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
        }
    }

    setConnectionStatus(connected) {
        const el = this.elements.connectionStatus;
        el.classList.toggle('connected', connected);
        el.classList.toggle('disconnected', !connected);
        el.querySelector('.text').textContent = connected ? 'Connected' : 'Disconnected';
    }

    handleMessage(message) {
        switch (message.type) {
            case 'state':
                this.updateState(message.data);
                break;
            case 'stats':
                this.stats = message.data;
                this.updateStats(message.data);
                this.updateModeIndicator(message.data);
                break;
            case 'project_change':
                this.controlsRendered = false;
                break;
        }
    }

    updateState(state) {
        this.state = state;

        // Update project list
        this.renderProjectList(state.available_projects || []);

        // Update current project panel
        if (state.current_project && !state.menu_visible) {
            this.elements.projectSelector.classList.add('hidden');
            this.elements.currentProject.classList.remove('hidden');
            this.elements.projectName.textContent = state.current_project.name;

            // Render controls once when project is selected
            if (!this.controlsRendered) {
                this.renderDefaultControls();
                this.controlsRendered = true;
            }
        } else {
            this.elements.projectSelector.classList.remove('hidden');
            this.elements.currentProject.classList.add('hidden');
            this.controlsRendered = false;
        }
    }

    renderProjectList(projects) {
        const container = this.elements.projectList;
        container.innerHTML = '';

        projects.forEach(project => {
            const card = document.createElement('div');
            card.className = 'project-card';
            if (this.state?.current_project?.name === project.name) {
                card.classList.add('active');
            }

            card.innerHTML = `
                <h3>${project.name}</h3>
                <p>${project.description}</p>
            `;

            card.addEventListener('click', () => {
                this.sendCommand('select_project', { project_id: project.id });
            });

            container.appendChild(card);
        });
    }

    updateModeIndicator(stats) {
        const modeEl = this.elements.projectMode;
        if (!modeEl) return;

        const isTraining = stats.training_mode === true;
        modeEl.textContent = isTraining ? 'Training Mode' : 'Watch Mode';
        modeEl.className = 'mode-badge ' + (isTraining ? 'training' : 'watch');

        // Update toggle state if it exists
        const toggle = document.querySelector('[data-id="training_mode"]');
        if (toggle) {
            toggle.classList.toggle('active', isTraining);
        }
    }

    updateStats(stats) {
        const container = this.elements.statsPanel;
        container.innerHTML = '';

        const statConfig = {
            episode: { label: 'Episode' },
            best_score: { label: 'Best Score', highlight: true },
            avg_score_100: { label: 'Avg Score' },
            epsilon: { label: 'Epsilon' },
            current_score: { label: 'Current', highlight: true },
            memory_size: { label: 'Memory' },
        };

        Object.entries(stats).forEach(([key, value]) => {
            const config = statConfig[key];
            if (!config) return;

            const item = document.createElement('div');
            item.className = 'stat-item';
            if (config.highlight) {
                item.classList.add('highlight');
            }

            // Format value
            let displayValue = value;
            if (typeof value === 'number') {
                if (Number.isInteger(value)) {
                    displayValue = value.toLocaleString();
                } else {
                    displayValue = value.toFixed(3);
                }
            }

            item.innerHTML = `
                <div class="label">${config.label}</div>
                <div class="value">${displayValue}</div>
            `;

            container.appendChild(item);
        });
    }

    renderDefaultControls() {
        // Render the standard Flappy RL controls
        const controls = [
            { type: 'toggle', id: 'training_mode', label: 'Training Mode', value: false },
            { type: 'toggle', id: 'pause', label: 'Pause', value: false },
            { type: 'slider', id: 'speed', label: 'Speed', value: 1, min_value: 1, max_value: 50, step: 1 },
            { type: 'toggle', id: 'show_network', label: 'Show Network', value: false },
            { type: 'button', id: 'save', label: 'Save Model' },
            { type: 'button', id: 'load', label: 'Load Best' },
            { type: 'button', id: 'reset', label: 'Reset Training' },
        ];

        this.renderControls(controls);
    }

    renderControls(controls) {
        const container = this.elements.controlsPanel;
        container.innerHTML = '';

        controls.forEach(control => {
            const el = this.createControl(control);
            if (el) {
                container.appendChild(el);
            }
        });
    }

    createControl(control) {
        switch (control.type) {
            case 'button':
                const btn = document.createElement('button');
                btn.className = 'btn';
                if (control.id === 'reset') {
                    btn.className = 'btn btn-danger';
                }
                btn.textContent = control.label;
                btn.addEventListener('click', () => {
                    if (control.id === 'reset') {
                        if (confirm('Reset all training progress?')) {
                            this.sendCommand(control.id);
                        }
                    } else {
                        this.sendCommand(control.id);
                    }
                });
                return btn;

            case 'toggle':
                const toggleDiv = document.createElement('div');
                toggleDiv.className = 'toggle-control';

                // Special styling for training mode toggle
                if (control.id === 'training_mode') {
                    toggleDiv.classList.add('featured');
                }

                toggleDiv.innerHTML = `
                    <label>${control.label}</label>
                    <div class="toggle ${control.value ? 'active' : ''}" data-id="${control.id}"></div>
                `;
                toggleDiv.querySelector('.toggle').addEventListener('click', (e) => {
                    const toggle = e.target;
                    const active = !toggle.classList.contains('active');
                    toggle.classList.toggle('active', active);
                    this.sendCommand(control.id, { value: active });
                });
                return toggleDiv;

            case 'slider':
                const sliderDiv = document.createElement('div');
                sliderDiv.className = 'slider-control';
                sliderDiv.innerHTML = `
                    <label>
                        <span>${control.label}</span>
                        <span class="slider-value">${control.value}</span>
                    </label>
                    <input type="range"
                           min="${control.min_value}"
                           max="${control.max_value}"
                           step="${control.step}"
                           value="${control.value}">
                `;
                const input = sliderDiv.querySelector('input');
                const valueDisplay = sliderDiv.querySelector('.slider-value');
                input.addEventListener('input', (e) => {
                    const value = parseFloat(e.target.value);
                    valueDisplay.textContent = value;
                    this.sendCommand(control.id, { value });
                });
                return sliderDiv;

            default:
                return null;
        }
    }

    sendCommand(command, data = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'command',
                command,
                data
            }));
        }
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.controller = new DisplayController();
});
