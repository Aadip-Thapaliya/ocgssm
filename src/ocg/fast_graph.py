"""S5b fast post-switch graph tracking: EMA reset + replay-buffer structural updates.

The detector still scores the current single window exactly as ``GraphLearner.online_update``
does. Only the structural optimizer step changes: for ``replay_steps`` updates after a
confirmed detection the loss is computed on the most recent ``buffer_size`` observed input
windows (the current window included; no future windows, no forecast targets).
"""

from collections import deque

import torch
from torch import Tensor

from ocg.graph import GraphLearner


def enable_fast_tracking(learner: GraphLearner, buffer_size: int, replay_steps: int,
                         reset_ema: bool = True) -> GraphLearner:
    """Switch an existing learner (with its pretrained state) to the fast-tracking update."""
    if buffer_size < 1 or replay_steps < 1:
        raise ValueError("buffer_size and replay_steps must be positive")
    learner.__class__ = FastGraphLearner
    learner.fast_buffer = deque(maxlen=buffer_size)
    learner.fast_replay_steps = replay_steps
    learner.fast_reset_ema = reset_ema
    learner._fast_remaining = 0
    learner.fast_events = []
    return learner


class FastGraphLearner(GraphLearner):
    """GraphLearner whose post-detection structural steps use a replay buffer."""

    def reset_ema_to_current(self) -> None:
        """Set ``ema_logits`` (the buffer used by ``adjacency(use_ema=True)``) to ``logits``."""
        with torch.no_grad():
            self.ema_logits.copy_(self.logits.detach())

    def online_update(self, values: Tensor, update_detector: bool = True) -> float:
        self._steps += 1
        for window in values.detach():
            self.fast_buffer.append(window.clone())
        loss = self.structural_loss(values)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"online structural graph loss non-finite at step {self._steps}")
        outlier = self.detector.update(loss.detach(), update_statistics=update_detector)
        if outlier:
            self._detection_streak += 1
            self._normal_streak = 0
        else:
            self._detection_streak = 0
            self._normal_streak += 1
            if self._normal_streak >= self.detector_rearm_steps:
                self._detector_armed = True
        self.last_detection = (self._detector_armed
                               and self._detection_streak >= self.detector_confirm_steps)
        if self.last_detection:
            self._detector_armed = False
            self._normal_streak = 0
            self._boost_remaining = self.boost_steps
            self._fast_remaining = self.fast_replay_steps
            if self.fast_reset_ema:
                self.reset_ema_to_current()
            self.fast_events.append(self._steps)
        if self._steps % self.graph_stride == 0:
            learning_rate = self._base_learning_rate
            if self._boost_remaining > 0:
                learning_rate *= self.boost
            for group in self._optimizer.param_groups:
                group["lr"] = learning_rate
            step_loss = loss
            if self._fast_remaining > 0:
                step_loss = self.structural_loss(torch.stack(list(self.fast_buffer)))
                self._fast_remaining -= 1
            self._optimizer.zero_grad()
            step_loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in self.parameters()):
                raise FloatingPointError(f"online structural gradient non-finite at step {self._steps}")
            self._optimizer.step()
            self._update_ema()
            self._boost_remaining = max(0, self._boost_remaining - 1)
        return float(loss.detach().item())
