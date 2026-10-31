"""S1 direct-head modes: companion reachability, iterated rollout, defaults."""

import torch

from ocg.data import generate_svar, make_windows
from ocg.model import CausalSSM, companion_reachability


def _graph(k: int, d: int, edges: list[tuple[int, int, int]]) -> torch.Tensor:
    graph = torch.zeros(k, d, d)
    for lag_index, target, source in edges:
        graph[lag_index, target, source] = 1.0
    return graph


def test_companion_orientation_and_k2_h2_blocks() -> None:
    # A1: 0 -> 1, A2: 1 -> 2 (entry [target, source]).
    graph = _graph(2, 3, [(0, 1, 0), (1, 2, 1)])
    masks = companion_reachability(graph, 3)
    assert masks.shape == (3, 2, 3, 3)
    assert torch.equal(masks[0], graph)
    a1, a2 = graph[0], graph[1]
    assert torch.equal(masks[1, 0], ((a1 @ a1 + a2) > 0).float())
    assert torch.equal(masks[1, 1], ((a1 @ a2) > 0).float())
    # x2[t+2] depends on x1[t] through A2 (lag 2 from t+2), and x1[t] on x0[t-1].
    assert masks[1, 0, 2, 1] == 1 and masks[1, 0, 1, 0] == 0
    # H3, lag 1: x2[t+3] <- x1[t+1] <- x0[t] via A2 then A1.
    assert masks[2, 0, 2, 0] == 1


def test_reachability_contains_true_multistep_coefficient_support() -> None:
    torch.manual_seed(0)
    graph = (torch.rand(2, 6, 6) < 0.3).float()
    coefficients = graph * (torch.rand(2, 6, 6) * 0.4 + 0.1)
    size = 12
    companion = torch.zeros(size, size, dtype=torch.float64)
    companion[:6] = torch.cat(tuple(coefficients.double()), dim=1)
    companion[6:, :6] = torch.eye(6, dtype=torch.float64)
    masks = companion_reachability(graph, 4)
    power = torch.eye(size, dtype=torch.float64)
    for h in range(4):
        power = power @ companion
        for k in range(2):
            block = power[:6, 6 * k : 6 * (k + 1)]
            assert torch.all(masks[h, k][block.abs() > 0] == 1)
            # Positive coefficients: support matches exactly.
            assert torch.equal((block > 0).float(), masks[h, k])


def _oracle_model(mode: str, graph: torch.Tensor) -> CausalSSM:
    torch.manual_seed(3)
    return CausalSSM(
        n_variables=graph.shape[1], state_dim=4, horizon=4, n_lags=graph.shape[0],
        embedding_dim=4, graph_mode="oracle", oracle_graph=graph, direct_head_mode=mode,
    )


def test_iterated_rollout_matches_manual_recursion() -> None:
    graph = _graph(2, 3, [(0, 0, 0), (0, 1, 0), (1, 2, 1), (0, 2, 2)])
    model = _oracle_model("iterated", graph)
    with torch.no_grad():
        model.lag_forecast_weights.normal_()
    values = torch.randn(5, 8, 3)
    got = model._direct_lag_forecast(values)
    w1 = model.lag_forecast_weights[0, 0] * graph[0]
    w2 = model.lag_forecast_weights[0, 1] * graph[1]
    previous, current = values[:, -2], values[:, -1]
    expected = []
    for _ in range(4):
        nxt = current @ w1.T + previous @ w2.T
        expected.append(nxt)
        previous, current = current, nxt
    assert torch.allclose(got, torch.stack(expected, 1), atol=1e-6)


def test_reachability_head_uses_horizon_masks() -> None:
    graph = _graph(2, 3, [(0, 1, 0), (1, 2, 1)])
    model = _oracle_model("reachability", graph)
    with torch.no_grad():
        model.lag_forecast_weights.fill_(1.0)
    values = torch.zeros(1, 8, 3)
    values[0, -1, 1] = 1.0  # x1[t]
    out = model._direct_lag_forecast(values)
    assert out[0, 0].tolist() == [0.0, 0.0, 0.0]  # H1 lag-1 has no 1->2 edge
    assert out[0, 1, 2] == 1.0  # H2 reaches x2 via A2


def test_default_mode_is_unchanged_and_zero_head_gives_direct_var() -> None:
    graph = _graph(2, 3, [(0, 1, 0), (1, 2, 1)])
    values = torch.randn(2, 8, 3)
    torch.manual_seed(5)
    default = CausalSSM(3, 4, 4, 2, embedding_dim=4, graph_mode="oracle", oracle_graph=graph)
    torch.manual_seed(5)
    lag = CausalSSM(
        3, 4, 4, 2, embedding_dim=4, graph_mode="oracle", oracle_graph=graph,
        direct_head_mode="lag",
    )
    assert torch.equal(default(values), lag(values))
    zero = CausalSSM(
        3, 4, 4, 2, embedding_dim=4, graph_mode="oracle", oracle_graph=graph,
        zero_init_ssm_head=True,
    )
    _, parts = zero.forward_with_components(values)
    assert torch.count_nonzero(parts["ssm_contribution"]) == 0


def test_s1_split_boundaries_exclude_test_labels() -> None:
    data = generate_svar(
        n_steps=4000, n_variables=4, n_lags=2, switch_at=3600, blend_steps=0,
        sparsity=0.1, noise_scale=0.1, seed=1,
    )
    series = data["series"]
    index = torch.arange(4000, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    for start, stop in ((0, 2400), (2400, 2666), (2666, 2933), (2933, 3200)):
        x, y = make_windows(index[start:stop], 24, 6)
        assert int(y.max()) <= 3199 and int(x.min()) >= start
        assert int(y.max()) == stop - 1
    assert series.shape[0] == 4000
