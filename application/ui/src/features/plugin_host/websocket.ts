/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Simple event emitter for browser environment
 * Replaces Node.js EventEmitter
 */
class EventBus {
    private listeners: Map<string, Set<(data: any) => void>> = new Map();

    on(event: string, callback: (data: any) => void): void {
        if (!this.listeners.has(event)) {
            console.log('Adding listener for event:', event);
            this.listeners.set(event, new Set());
        }
        console.log('Adding callback to listener for event:', event);
        this.listeners.get(event)!.add(callback);
        console.log('Listeners:', this.listeners);
    }

    off(event: string, callback: (data: any) => void): void {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(callback);
        }
    }

    emit(event: string, data: any): void {
        const eventListeners = this.listeners.get(event);
        console.log('Emitting event:', event, 'with data:', data);
        if (eventListeners) {
            eventListeners.forEach((callback) => callback(data));
        }
    }

    removeAllListeners(event?: string): void {
        if (event) {
            this.listeners.delete(event);
        } else {
            this.listeners.clear();
        }
    }
}

export const eventBus = new EventBus();

let ws: WebSocket | null = null;

/**
 * Connect to WebSocket server (browser WebSocket client)
 * @param url WebSocket server URL (e.g., 'ws://localhost:8080')
 */
export function connectWebSocket(url: string = 'ws://localhost:8000') {
    if (ws?.readyState === WebSocket.OPEN) {
        console.warn('WebSocket already connected');
        return ws;
    }

    console.debug(`Connecting to WebSocket server at ${url}...`);

    try {
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('✅ WebSocket connected');
        };

        ws.onmessage = (event: MessageEvent) => {
            try {
                const data = JSON.parse(event.data);
                console.debug('WebSocket message received:', data);

                // Emit event through event bus for plugins to listen
                if (data.event && data.data !== undefined) {
                    eventBus.emit(data.event, data.data);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', event.data, error);
            }
        };

        ws.onerror = (error: Event) => {
            console.warn('⚠️ WebSocket connection failed. Make sure the backend server is running on ' + url);
            console.debug('WebSocket error details:', error);
        };

        ws.onclose = (event: CloseEvent) => {
            console.log('WebSocket disconnected', event.code === 1000 ? '' : `(code: ${event.code})`);
            ws = null;
        };
    } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        ws = null;
    }

    return ws;
}

/**
 * Disconnect from WebSocket server
 */
export function disconnectWebSocket() {
    if (ws) {
        console.debug('Disconnecting WebSocket...');
        ws.close();
        ws = null;
    }
}

/**
 * Send message to WebSocket server
 */
export function sendWebSocketMessage(event: string, data: any) {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event, data }));
    } else {
        console.warn('WebSocket not connected, cannot send message');
    }
}
