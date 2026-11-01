import numpy as np
import pytest
import torch

from ocg.baselines import DirectVAR
from scripts import roadmap_final as rf


def test_iterated_direct_var_rolls_one_step_map():
    torch.manual_seed(0)
    model = DirectVAR(n_lags=2, horizon=3, n_variables=4, iterated=True)
    torch.nn.init.normal_(model.projection.weight, std=0.3)
    x = torch.randn(5, 10, 4)
    out = model(x)
    a1, a2 = model.projection.weight[:, :4], model.projection.weight[:, 4:]
    b = model.projection.bias
    h1 = x[:, -1] @ a1.T + x[:, -2] @ a2.T + b
    h2 = h1 @ a1.T + x[:, -1] @ a2.T + b
    h3 = h2 @ a1.T + h1 @ a2.T + b
    assert torch.allclose(out, torch.stack([h1, h2, h3], dim=1), atol=1e-6)
    assert out.shape == (5, 3, 4)


def test_development_data_never_reads_test_targets():
    data = rf.synthetic_data((67, 1009), variables=5, train_length=600)
    assert "test" not in data
    assert data["gate_target_start"] == 2933 + 24
    assert data["segments"]["train"] == (1800, 2400)
    x, y, _ = data["gate"]
    assert data["gate_target_start"] + len(x) - 1 + 5 == 3199


def test_held_out_and_burned_pairs_are_guarded():
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD"):
        rf.synthetic_data((53, 709), variables=5, split="test")
    with pytest.raises(RuntimeError, match="not a development pair"):
        rf.synthetic_data((53, 709), variables=5)
    with pytest.raises(RuntimeError, match="burned"):
        rf.synthetic_data((31, 301), variables=5)


def test_paired_statistics():
    rng = np.random.default_rng(0)
    a = [rng.random(200) + 1.0 for _ in range(3)]
    b = [x * 2.0 for x in a]
    ratio = rf.paired_ratio(a, b, 1, resamples=200)
    assert ratio["ratio"] == pytest.approx(0.5)
    assert ratio["ci95"][0] == pytest.approx(0.5) and ratio["ci95"][1] == pytest.approx(0.5)
    diff = rf.paired_difference(a, b, 1, resamples=200)
    assert diff["difference"] < 0 and diff["ci95"][1] < 0


def test_fixed_mask_and_random_density_builds():
    graph0 = torch.zeros(2, 5, 5)
    mask = torch.eye(5).repeat(2, 1, 1)
    model = rf.build({"graph_mode": "fixed_mask"}, 5, 1, graph0, mask)
    assert torch.equal(model.fixed_graph, mask)
    rnd = rf.build({"graph_mode": "random", "random_density": 0.2}, 5, 1, graph0)
    assert int(rnd.fixed_graph[0].sum()) == 5


def test_s5_windows_and_guards():
    from scripts import roadmap_s5 as s5
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD"):
        s5.s5_data((79, 1301))
    data = s5.s5_data((67, 1009))
    sx, sy, sg = data["stream"]
    assert len(sx) == 971
    first_post_target = data["stream_target_start"] + s5.FIRST_POST
    assert first_post_target == s5.SWITCH
    assert data["stream_target_start"] + s5.LAST_PRE + 5 == s5.SWITCH - 1
    assert torch.equal(sg[s5.DETECT_INDEX - 1], data["graph0"])
    assert torch.equal(sg[s5.DETECT_INDEX], data["graph1"])
    assert data["tune_target_start"] + len(data["tune"][0]) - 1 + 5 == 1999


def test_s6_split_and_train_only_standardization():
    from pathlib import Path
    from scripts import roadmap_s6 as s6
    if not (Path("data") / "ETTh1.csv").exists():
        pytest.skip("ETTh1.csv not downloaded")
    d = s6.ett_data("ETTh1", 24)
    assert d["train_target_range"][1] < s6.TRAIN_END
    assert d["val_target_range"] == [s6.TRAIN_END, s6.VAL_END - 1]
    assert d["test_target_range"] == [s6.VAL_END, s6.TEST_END - 1]
    raw = s6.load_csv(Path("data") / "ETTh1.csv", s6.COLUMNS)[: s6.TRAIN_END]
    assert torch.allclose(torch.tensor(d["mean"]), raw.float().mean(dim=0))
    assert d["test"][0].shape[0] == s6.TEST_END - s6.VAL_END - 24 + 1
