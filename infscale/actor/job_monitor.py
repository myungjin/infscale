# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from multiprocessing import connection

import torch.multiprocessing as mp
from infscale import get_logger

logger = get_logger()


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process


class JobMonitor:
    """JobMonitor class."""

    def __init__(self, metadata: WorkerMetaData):
        self.metadata = metadata

    def start_monitoring(self):
        # TODO: this loop should be revised in an event-driven fashion
        for rank, worker in self.metadata.items():
            message = worker.pipe.recv()
            print(f"received message: {message} from rank: {rank}")
