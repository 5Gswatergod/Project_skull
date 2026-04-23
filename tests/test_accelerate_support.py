from __future__ import annotations

from unittest.mock import patch

import pytest

from skull.train import accelerate_support


def test_build_accelerator_returns_none_when_disabled():
    assert accelerate_support.build_accelerator({}) is None


def test_build_accelerator_raises_clear_error_when_missing_dependency():
    with patch.object(accelerate_support, "Accelerator", None):
        with pytest.raises(ImportError, match="accelerate"):
            accelerate_support.build_accelerator({"use_accelerate": True})


def test_normalize_mixed_precision_defaults_to_no():
    assert accelerate_support.normalize_mixed_precision("fp16") == "fp16"
    assert accelerate_support.normalize_mixed_precision("bf16") == "bf16"
    assert accelerate_support.normalize_mixed_precision("fp32") == "no"
