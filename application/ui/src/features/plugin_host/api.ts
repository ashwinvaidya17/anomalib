/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type * as geti_inspect from 'geti-inspect';

import { eventBus, sendWebSocketMessage } from './websocket';

export interface PluginContext {
    /**
     * Array to which disposables can be added.
     */
    subscriptions: {
        // Function to clean up resources.
        dispose(): void;
    }[];
}

export interface PluginAPI {
    onEvent(event: string, callback: (data: any) => void): { dispose: () => void };
    sendEvent(event: string, data?: any): void;
    window: {
        createStatusBarItem(
            id: string,
            alignment: geti_inspect.StatusBarItemAlignment,
            priority: number
        ): geti_inspect.StatusBarItem;
    };
}

export const api: PluginAPI = {
    onEvent: (event: string, callback: (data: any) => void) => {
        eventBus.on(event, callback);
        // Return disposable to allow cleanup
        return {
            dispose: () => {
                eventBus.off(event, callback);
            },
        };
    },
    sendEvent: (event: string, data?: any) => {
        console.log('Sending event:', event, data);
        sendWebSocketMessage(event, data);
    },
    window: {
        createStatusBarItem: (
            id: string,
            alignment: geti_inspect.StatusBarItemAlignment,
            priority: number
        ): geti_inspect.StatusBarItem => {
            const item: geti_inspect.StatusBarItem = {
                id,
                alignment,
                priority,
                element: '',
                dispose: () => {
                    // Dispose will be overridden by PluginProvider to remove from state
                    console.debug(`StatusBarItem ${id} disposed`);
                },
            };
            return item;
        },
    },
};
