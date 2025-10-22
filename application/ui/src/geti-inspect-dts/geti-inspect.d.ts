/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Intel Corporation. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *--------------------------------------------------------------------------------------------*/

/**
 * Note: Code in this file is inspired by the VS Code API.
 */

declare module 'geti-inspect' {
    export enum StatusBarItemAlignment {
        Left = 1,
        Right = 2,
    }

    export interface StatusBarItem {
        readonly id: string;
        readonly alignment: StatusBarItemAlignment;
        /**
         * The priority of the item. Higher value means the item should be placed more to the left.
         */
        readonly priority: number;
        // Element to show for entry
        element: React.ReactNode;

        dispose(): void;
    }

    export function onEvent(event: string, callback: (data: any) => void): void;
    export function sendEvent(event: string, data: any): void;
    export namespace window {
        export function createStatusBarItem(
            id: string,
            alignment: StatusBarItemAlignment,
            priority: number
        ): StatusBarItem;
    }
}
