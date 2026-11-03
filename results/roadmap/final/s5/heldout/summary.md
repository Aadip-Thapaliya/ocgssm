# S5 adaptation (heldout)

Gates commit `df795691c1fced8da127a764e6ece738574e0041`. Pairs: [[79, 1301], [83, 1409], [89, 1511], [97, 1613], [101, 1709]]. Gate: **FAIL** - learned_det_on < learned_frozen (CI excl 0): pass, learned_det_on < dense_online (CI excl 0): FAIL

Primary metric: mean MSE over the first 200 windows whose first target is post-switch.

| Variant | post-switch MSE | per seed | median window | pre-switch MSE | forecast recovery (windows) |
|---|---:|---|---:|---:|---|
| identity_online | 0.01675 | 0.01584, 0.01585, 0.01621, 0.01906, 0.01680 | 0.01620 | 0.01854 | [0, 0, 0, 0, 19] |
| dense_online | 0.01716 | 0.01617, 0.01698, 0.01639, 0.01955, 0.01673 | 0.01644 | 0.01618 | [15, 0, 0, 33, 38] |
| random_online | 0.01688 | 0.01596, 0.01611, 0.01629, 0.01911, 0.01693 | 0.01633 | 0.01806 | [0, 0, 0, 0, 23] |
| oracle_online | 0.01558 | 0.01449, 0.01497, 0.01524, 0.01771, 0.01548 | 0.01490 | 0.01554 | [0, 0, 0, 31, 26] |
| learned_frozen | 0.01894 | 0.01742, 0.01951, 0.01780, 0.02168, 0.01829 | 0.01827 | 0.01538 | [32, 0, 2, 37, 123] |
| learned_det_off | 0.01748 | 0.01632, 0.01713, 0.01661, 0.02019, 0.01712 | 0.01673 | 0.01554 | [17, 0, 0, 36, 40] |
| learned_det_on | 0.01728 | 0.01615, 0.01676, 0.01649, 0.02003, 0.01697 | 0.01663 | 0.01554 | [16, 0, 0, 35, 40] |

| Difference (post-switch MSE) | mean | 95% CI | per seed |
|---|---:|---|---|
| learned_det_on-identity_online | 0.000527 | [0.000371, 0.000678] | 0.000312, 0.000900, 0.000275, 0.000974, 0.000176 |
| learned_det_on-dense_online | 0.000117 | [0.000026, 0.000211] | -0.000020, -0.000229, 0.000099, 0.000487, 0.000247 |
| learned_det_on-random_online | 0.000399 | [0.000281, 0.000534] | 0.000192, 0.000641, 0.000197, 0.000923, 0.000042 |
| learned_det_on-oracle_online | 0.001704 | [0.001537, 0.001883] | 0.001658, 0.001787, 0.001254, 0.002323, 0.001497 |
| learned_det_on-learned_frozen | -0.001661 | [-0.001831, -0.001502] | -0.001276, -0.002757, -0.001306, -0.001650, -0.001315 |
| learned_det_on-learned_det_off | -0.000196 | [-0.000230, -0.000166] | -0.000176, -0.000377, -0.000124, -0.000161, -0.000141 |

learned_det_off: detector FA=0 hit=False delay=None; FA=0 hit=False delay=None; FA=0 hit=False delay=None; FA=0 hit=False delay=None; FA=0 hit=False delay=None

learned_det_off: graph preF1=0.188 post200F1=0.015 finalF1=0.000 topkF1 pre/post200/final=1.0/0.1724285714285714/0.37142857142857144 AUROC=0.853 SHD=35 rec=0; preF1=0.664 post200F1=0.034 finalF1=0.049 topkF1 pre/post200/final=0.997742299629092/0.25699999999999995/0.375 AUROC=0.860 SHD=39 rec=43; preF1=0.505 post200F1=0.025 finalF1=0.000 topkF1 pre/post200/final=1.0/0.31072916666666667/0.5208333333333334 AUROC=0.859 SHD=48 rec=0; preF1=0.627 post200F1=0.083 finalF1=0.000 topkF1 pre/post200/final=1.0/0.1909375/0.34375 AUROC=0.876 SHD=32 rec=34; preF1=0.110 post200F1=0.022 finalF1=0.053 topkF1 pre/post200/final=0.9974842767295597/0.1940540540540541/0.3783783783783784 AUROC=0.846 SHD=36 rec=0

learned_det_on: detector FA=0 hit=True delay=10; FA=0 hit=True delay=2; FA=0 hit=True delay=3; FA=0 hit=True delay=2; FA=0 hit=True delay=12

learned_det_on: graph preF1=0.188 post200F1=0.009 finalF1=0.000 topkF1 pre/post200/final=1.0/0.23299999999999998/0.45714285714285713 AUROC=0.884 SHD=35 rec=0; preF1=0.664 post200F1=0.019 finalF1=0.000 topkF1 pre/post200/final=0.997742299629092/0.33349999999999996/0.6 AUROC=0.883 SHD=40 rec=15; preF1=0.505 post200F1=0.012 finalF1=0.000 topkF1 pre/post200/final=1.0/0.37249999999999994/0.6458333333333334 AUROC=0.877 SHD=48 rec=0; preF1=0.627 post200F1=0.039 finalF1=0.000 topkF1 pre/post200/final=1.0/0.23515625/0.375 AUROC=0.867 SHD=32 rec=22; preF1=0.110 post200F1=0.010 finalF1=0.000 topkF1 pre/post200/final=0.9974842767295597/0.23202702702702704/0.5135135135135135 AUROC=0.887 SHD=37 rec=0
