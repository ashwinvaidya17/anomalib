import React from 'react';

import { Flex, ProgressBar, Text } from '@geti/ui';
import { CanceledIcon, WaitingIcon } from '@geti/ui/icons';

export function IdleItem(): React.ReactNode {
    return (
        <Flex
            direction='row'
            alignItems='center'
            width='100px'
            justifyContent='space-between'
            UNSAFE_style={{ padding: '5px' }}
        >
            <WaitingIcon height='14px' width='14px' stroke='var(--spectrum-global-color-gray-600)' />
            <Text marginStart={'5px'} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                Idle
            </Text>
        </Flex>
    );
}

export function TrainingStatusItem(progress: number, stage: string, onCancel?: () => void): React.ReactNode {
    // Determine color based on stage
    let bgcolor = 'var(--spectrum-global-color-blue-600)';
    let fgcolor = '#fff';
    if (stage.toLowerCase().includes('valid')) {
        bgcolor = 'var(--spectrum-global-color-yellow-600)';
        fgcolor = '#000';
    } else if (stage.toLowerCase().includes('test')) {
        bgcolor = 'var(--spectrum-global-color-green-600)';
        fgcolor = '#fff';
    } else if (stage.toLowerCase().includes('train') || stage.toLowerCase().includes('fit')) {
        bgcolor = 'var(--spectrum-global-color-blue-600)';
        fgcolor = '#fff';
    }

    return (
        <div
            style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: bgcolor,
            }}
        >
            <Flex direction='row' alignItems='center' width='100px' justifyContent='space-between'>
                <button
                    onClick={() => {
                        console.log('Cancel training');
                        if (onCancel) {
                            onCancel();
                        }
                    }}
                    style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                    }}
                >
                    <CanceledIcon height='14px' width='14px' stroke={fgcolor} />
                </button>
                <Text
                    UNSAFE_style={{
                        fontSize: '12px',
                        marginBottom: '4px',
                        marginRight: '4px',
                        textAlign: 'center',
                        color: fgcolor,
                    }}
                >
                    {stage}
                </Text>
            </Flex>
            <ProgressBar value={progress} aria-label={stage} width='100px' showValueLabel={false} />
        </div>
    );
}
