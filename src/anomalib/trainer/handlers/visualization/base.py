"""Base class for visualization handlers."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod


class VisualizationHandler:
    @abstractmethod
    def setup(self):
        """Sets up the handler."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Updates the handler.

        Affects internal state.
        """
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Computes the handler.

        Affects internal state.
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        """Processes the values passed to the handler."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the handler."""
        pass
