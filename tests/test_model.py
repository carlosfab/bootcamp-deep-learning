"""Testes basicos para verificar que o projeto esta funcional."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import LunaModel


def test_model_forward_shape():
    """Verifica que o modelo recebe um crop e retorna as shapes corretas."""
    model = LunaModel()
    model.eval()
    batch = torch.randn(2, 1, 32, 48, 48)
    with torch.no_grad():
        logits, probs = model(batch)
    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)


def test_model_probs_sum_to_one():
    """Verifica que as probabilidades somam 1 (softmax)."""
    model = LunaModel()
    model.eval()
    batch = torch.randn(4, 1, 32, 48, 48)
    with torch.no_grad():
        _, probs = model(batch)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_model_parameter_count():
    """Verifica que o modelo tem o numero esperado de parametros."""
    model = LunaModel()
    total = sum(p.numel() for p in model.parameters())
    assert total == 222_220


def test_checkpoint_loads():
    """Verifica que o checkpoint carrega e produz output valido."""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "luna_model_best.pt"
    if not ckpt_path.exists():
        import pytest
        pytest.skip("Checkpoint nao encontrado")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "epoch" in ckpt

    model = LunaModel()
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = torch.randn(1, 1, 32, 48, 48)
    with torch.no_grad():
        logits, probs = model(batch)
    assert probs.shape == (1, 2)
    assert 0 <= probs[0, 1].item() <= 1
