import React from 'react';

import ReactDOM from 'react-dom/client';

import { connectWebSocket } from './features/plugin_host/websocket';
import { Providers } from './providers';

import './index.css';

async function init() {
    console.debug('Initializing application...');
    // Connect to WebSocket server (backend FastAPI server)
    connectWebSocket('ws://localhost:8000/api/ws/');
    // NOTE: Plugins are now loaded by PluginProvider, not here
    // TODO: websocket connection should be refreshed periodically in case it drops
}

init();

const rootEl = document.getElementById('root');
if (rootEl) {
    const root = ReactDOM.createRoot(rootEl);
    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
