# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

from vllm.entrypoints import grpc_server


def test_grpc_server_options_honor_max_ping_strikes_env():
    assert hasattr(grpc_server, "_grpc_server_options")

    with patch.dict(
        "os.environ",
        {"SMG_GRPC_MAX_PING_STRIKES": "0"},
        clear=False,
    ):
        options = dict(grpc_server._grpc_server_options())

    assert options["grpc.http2.max_ping_strikes"] == 0


def test_grpc_server_options_allow_positive_strike_budget():
    assert hasattr(grpc_server, "_grpc_server_options")

    with patch.dict(
        "os.environ",
        {"SMG_GRPC_MAX_PING_STRIKES": "7"},
        clear=False,
    ):
        options = dict(grpc_server._grpc_server_options())

    assert options["grpc.http2.max_ping_strikes"] == 7
