from __future__ import annotations

from unittest.mock import patch

from skull.cli.eval import _resolve_device as resolve_eval_device
from skull.cli.sample import _resolve_device as resolve_sample_device
from skull.train.trainer_pretrain import _resolve_device as resolve_pretrain_device
from skull.train.trainer_sft import _resolve_device as resolve_sft_device


def test_device_falls_back_to_cpu_when_cuda_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_pretrain_device("cuda").type == "cpu"
        assert resolve_sft_device("cuda").type == "cpu"
        assert resolve_eval_device("cuda").type == "cpu"
        assert resolve_sample_device("cuda").type == "cpu"
