/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { api } from './api';

interface Plugin {
    activate?: (api: any) => void;
    deactivate?: () => void;
}

// Store loaded plugins for cleanup
const loadedPlugins: Plugin[] = [];

/**
 * Load all plugins and activate them
 */
export async function loadPlugins() {
    const pluginModules = [
        import('../../extensions/progressbar/index.ts'),
        // Add more plugins here
    ];

    for (const pluginPromise of pluginModules) {
        try {
            const plugin = await pluginPromise;

            if (plugin.activate) {
                // Check if plugin is already loaded (prevent double-activation in React Strict Mode)
                if (loadedPlugins.includes(plugin)) {
                    console.debug('Plugin already loaded, skipping activation:', plugin);
                    continue;
                }

                console.debug('Loading plugin:', plugin);
                plugin.activate(api);
                loadedPlugins.push(plugin);
            }
        } catch (error) {
            console.error('Failed to load plugin:', error);
        }
    }

    console.debug(`Loaded ${loadedPlugins.length} plugin(s)`);
}

/**
 * Unload all plugins and call their deactivate functions
 */
export function unloadPlugins() {
    console.debug('Unloading plugins...');

    for (const plugin of loadedPlugins) {
        try {
            if (plugin.deactivate) {
                plugin.deactivate();
            }
        } catch (error) {
            console.error('Error deactivating plugin:', error);
        }
    }

    loadedPlugins.length = 0; // Clear array
    console.debug('All plugins unloaded');
}
