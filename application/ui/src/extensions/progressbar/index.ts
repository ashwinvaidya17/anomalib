/** ---------------------------------------------------------------------------------------------
 *  Copyright (c) Intel Corporation. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *--------------------------------------------------------------------------------------------*/

import type * as geti_inspect from 'geti-inspect';

import { StatusBarItemAlignment } from '../../features/inspect/statusbar.component';
import type { PluginAPI } from '../../features/plugin_host/api';
import { IdleItem, TrainingStatusItem } from './components';

let progressbarItem: geti_inspect.StatusBarItem;
let eventDisposable: { dispose: () => void } | null = null;

function getElement(status: string, stage: string, progress: number, onCancel?: () => void): React.ReactNode {
    if (status === 'Running') {
        return TrainingStatusItem(progress, stage, onCancel);
    }
    return IdleItem();
}

export function activate(api: PluginAPI) {
    console.debug('Progressbar plugin activated');

    // Create cancel handler
    const handleCancel = () => {
        console.log('Sending cancel_training event to backend');
        api.sendEvent('cancel_training', {});
    };

    // Create status bar item
    progressbarItem = api.window.createStatusBarItem('progressbar', StatusBarItemAlignment.Right, 1);
    progressbarItem.element = getElement('Idle', 'Idle', 0);
    console.log('progressbarItem', progressbarItem);

    // Listen to progress events
    eventDisposable = api.onEvent('progress_update', (data: any) => {
        console.log('Progress update:', data);
        progressbarItem.element = getElement(data.status, data.stage, data.progress, handleCancel);
    });
}

/**
 * Plugin deactivation function - called when plugin unloads
 * Clean up all resources created by the plugin
 */
export function deactivate() {
    console.debug('Progressbar plugin deactivated');

    // Unsubscribe from events
    if (eventDisposable) {
        eventDisposable.dispose();
        eventDisposable = null;
    }

    // Dispose status bar item
    if (progressbarItem) {
        progressbarItem.dispose();
    }
}
