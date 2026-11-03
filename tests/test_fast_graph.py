import pytest
import torch

from ocg.fast_graph import FastGraphLearner, enable_fast_tracking
from ocg.graph import GraphLearner


def _learner() -> GraphLearner:
    torch.manual_seed(0)
    learner = GraphLearner(n_variables=4, n_lags=2, learning_rate=0.01, ema_decay=0.95)
    with torch.no_grad():
        learner.logits.add_(torch.randn_like(learner.logits))
    return learner


def test_reset_sets_ema_representation_to_current_logits() -> None:
    learner = enable_fast_tracking(_learner(), buffer_size=8, replay_steps=3)
    assert isinstance(learner, FastGraphLearner)
    assert not torch.allclose(learner.adjacency(use_ema=True), learner.adjacency(use_ema=False))
    learner.reset_ema_to_current()
    assert torch.equal(learner.adjacency(use_ema=True), learner.adjacency(use_ema=False))


def test_replay_uses_only_observed_windows_and_matches_base_without_detection() -> None:
    windows = torch.randn(12, 1, 24, 4)
    base, fast = _learner(), enable_fast_tracking(_learner(), buffer_size=5, replay_steps=3)
    for w in windows:
        assert base.online_update(w) == pytest.approx(fast.online_update(w))
    assert torch.allclose(base.logits, fast.logits)
    assert len(fast.fast_buffer) == 5
    assert torch.equal(fast.fast_buffer[-1], windows[-1, 0])


def test_detection_triggers_reset_and_replay_steps() -> None:
    fast = enable_fast_tracking(_learner(), buffer_size=4, replay_steps=2)
    flags = iter([False, True, False])
    fast.detector.update = lambda *a, **k: next(flags)
    fast.detector_rearm_steps = 1
    calm = torch.zeros(1, 24, 4)
    fast.online_update(calm)
    before = fast.logits.detach().clone()
    fast.online_update(torch.randn(1, 24, 4) * 50)
    assert fast.fast_events == [2]
    assert fast._fast_remaining == 1
    # EMA was reset to the pre-step logits, then moved by one EMA update.
    expected = 0.95 * before + 0.05 * fast.logits.detach()
    assert torch.allclose(fast.ema_logits, expected, atol=1e-6)
    fast.online_update(calm)
    assert fast._fast_remaining == 0
