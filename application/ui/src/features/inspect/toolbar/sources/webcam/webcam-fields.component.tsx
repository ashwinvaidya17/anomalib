// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { ActionButton, Flex, Item, Key, Loading, Picker, TextField } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { WebcamSourceConfig } from '../util';

type WebcamFieldsProps = {
    defaultState: WebcamSourceConfig;
};

export const WebcamFields = ({ defaultState }: WebcamFieldsProps) => {
    const { data: cameraDevices, isLoading, isRefetching, refetch } = $api.useQuery('get', '/api/devices/camera');
    const [name, setName] = useState(defaultState.name);
    // device id can be same for different backends hence we need to store the selected device id
    // Use unique key format: "index-backend"
    const [selectedDeviceKey, setSelectedDeviceKey] = useState<string>(`${defaultState.device_id}-${defaultState.backend}`);
    const [isModified, setIsModified] = useState(false);
    const [backend, setBackend] = useState(defaultState.backend);

    const devices = (cameraDevices?.devices ?? []).map((device) => ({
        id: `${device.index}-${device.backend}`, // Unique key combining index and backend
        index: device.index,
        name: device.name,
        backend: device.backend
    }));

    const handleNameChange = (value: string) => {
        setName(value);
        setIsModified(true);
    };

    const handleSelectionChange = (key: Key | null) => {
        if (key === null) {
            return;
        }
        console.log(key);

        const device = devices.find((d) => d.id === key);

        if (device) {
            // Update backend when device changes
            setBackend(device.backend);
            setSelectedDeviceKey(key as string);
            // if user modifies the name field, don't override it
            if (!isModified || !name?.trim()) {
                setName(device.name);
            }
        }
    };

    // Get the selected device's index for the form submission
    const selectedDevice = devices.find((d) => d.id === selectedDeviceKey);
    const deviceId = selectedDevice?.index ?? defaultState.device_id ?? 0;

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isHidden label='name' name='name' value={name} />
            <TextField isHidden label='backend' name='backend' value={backend.toString()} />
            <TextField isHidden label='device_id' name='device_id' value={deviceId.toString()} />
            <TextField width={'100%'} label='Name' name='name_display' value={name} onChange={handleNameChange} />

            <Flex alignItems='end' gap='size-200'>
                <Picker
                    flex='1'
                    label='Camera'
                    items={devices}
                    isLoading={isLoading}
                    defaultSelectedKey={selectedDeviceKey}
                    onSelectionChange={handleSelectionChange}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <ActionButton
                    onPress={() => refetch()}
                    isQuiet
                    aria-label='Refresh Cameras'
                    isDisabled={isLoading || isRefetching}
                >
                    {isRefetching ? <Loading mode={'inline'} size='S' /> : <Refresh />}
                </ActionButton>
            </Flex>
        </Flex>
    );
};
