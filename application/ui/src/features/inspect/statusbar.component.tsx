// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect } from 'react';

import { Flex, View } from '@geti/ui';

export enum StatusBarItemAlignment {
    Left = 1,
    Right = 2,
}

export interface StatusBarItem {
    id: string;
    alignment: StatusBarItemAlignment;
    priority: number;
    element: React.ReactNode;
}

export const StatusBar = ({ items }: { items: StatusBarItem[] }) => {
    useEffect(() => {
        return () => {
            console.log('StatusBarComponent unmounted');
        };
    }, []);

    const leftItems = items
        ? items.filter((item) => item.alignment === StatusBarItemAlignment.Left).sort((a, b) => b.priority - a.priority)
        : []; // Higher priority = more to the left

    const rightItems = items
        ? items
              .filter((item) => item.alignment === StatusBarItemAlignment.Right)
              .sort((a, b) => b.priority - a.priority)
        : []; // Higher priority = more to the left
    return (
        <View gridArea={'statusbar'} backgroundColor={'gray-100'} width={'100%'} height={'30px'} overflow={'hidden'}>
            <Flex direction={'row'} gap={'size-100'} justifyContent={'space-between'} width={'100%'}>
                {/* Left items */}

                <Flex direction={'row'} gap={'size-100'} justifyContent={'start'} maxHeight={'30px'}>
                    {leftItems.map((item) => (
                        <div key={item.id}>{item.element}</div>
                    ))}
                </Flex>

                {/* Right items */}

                <Flex direction={'row'} gap={'size-100'} justifyContent={'end'} maxHeight={'30px'}>
                    {rightItems.map((item) => (
                        <div key={item.id}>{item.element}</div>
                    ))}
                </Flex>
            </Flex>
        </View>
    );
};
