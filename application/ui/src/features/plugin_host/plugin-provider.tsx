/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import type { StatusBarItem } from '../inspect/statusbar.component';
import { api } from './api';
import { loadPlugins, unloadPlugins } from './loader';

interface PluginContextValue {
    statusBarItems: StatusBarItem[];
    isLoading: boolean;
}

const PluginContext = createContext<PluginContextValue | undefined>(undefined);

/**
 * Manages plugin lifecycle - loads plugins on mount, cleans up on unmount
 */
export const PluginProvider = ({ children }: { children: ReactNode }) => {
    const [statusBarItems, setStatusBarItems] = useState<StatusBarItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Use ref to store subscriptions so cleanup can access current value
    const subscriptionsRef = useRef<Array<{ dispose: () => void }>>([]);
    const originalApiRef = useRef(api.window.createStatusBarItem);

    useEffect(() => {
        console.debug('PluginProvider: Effect running...');

        // Store subscriptions for cleanup
        const subs: Array<{ dispose: () => void }> = [];

        // Override the createStatusBarItem to track items
        api.window.createStatusBarItem = (id, alignment, priority) => {
            const item = originalApiRef.current(id, alignment, priority);

            // Check for duplicates before adding
            setStatusBarItems((prev) => {
                // If item with this ID already exists, don't add it again
                if (prev.some((i) => i.id === id)) {
                    console.warn(`StatusBarItem with id '${id}' already exists, skipping duplicate`);
                    return prev;
                }
                return [...prev, item];
            });

            // Create proxy to track updates
            const proxy = new Proxy(item, {
                set(target: any, prop: string, value: any) {
                    target[prop] = value;
                    // Trigger re-render when element changes
                    if (prop === 'element') {
                        setStatusBarItems((prev) => [...prev]); // Force update
                    }
                    return true;
                },
            });

            // Override dispose to remove from state
            const originalDispose = proxy.dispose;
            proxy.dispose = () => {
                originalDispose();
                setStatusBarItems((prev) => prev.filter((i) => i.id !== id));
            };

            // Track for cleanup
            subs.push(proxy);

            return proxy;
        };

        // Load plugins
        loadPlugins()
            .then(() => {
                console.debug('PluginProvider: Plugins loaded');
                subscriptionsRef.current = subs;
                setIsLoading(false);
            })
            .catch((error) => {
                console.error('PluginProvider: Error loading plugins', error);
                setIsLoading(false);
            });

        // Cleanup on unmount
        return () => {
            console.debug('PluginProvider: Cleaning up plugins...');

            // Unload plugins (calls deactivate)
            unloadPlugins();

            // Dispose all subscriptions (using ref to get current value)
            subscriptionsRef.current.forEach((sub) => {
                if (sub.dispose) {
                    sub.dispose();
                }
            });

            // Clear subscriptions
            subscriptionsRef.current = [];

            // Clear status bar items
            setStatusBarItems([]);

            // Restore original API
            api.window.createStatusBarItem = originalApiRef.current;
        };
    }, []); // Empty deps - only run on mount/unmount

    return <PluginContext.Provider value={{ statusBarItems, isLoading }}>{children}</PluginContext.Provider>;
};

/**
 * Hook to access plugin context
 */
export const usePlugins = () => {
    const context = useContext(PluginContext);
    if (!context) {
        throw new Error('usePlugins must be used within a PluginProvider');
    }
    return context;
};

/**
 * Hook to get status bar items from plugins
 */
export const useStatusBarItems = () => {
    const { statusBarItems } = usePlugins();
    return statusBarItems;
};
