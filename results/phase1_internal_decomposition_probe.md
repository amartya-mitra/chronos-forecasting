# Phase 1 — Internal Decomposition Probe

**Question**: Does frozen Chronos-T5-Small already encode trend/seasonal/residual in its encoder hidden states? (No training, linear probes only.)

**Backbone**: `amazon/chronos-t5-small` pretrained, fully frozen, no prefix injection.

**Alignment**: context length L → seq_len L+1 (EOS appended). Position t=0..L-1 in hidden states aligns 1:1 with STL time step t. EOS excluded.

**Ridge probe**: `Ridge(alpha=1.0)`, features StandardScaled, series-level 80/20 split (`TRAIN_SEED={TRAIN_SEED}`).

## Aggregate R²_val (mean across 5 datasets)

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.0726 | 0.0729 | 0.0340 |
| 1 | -0.0500 | 0.1382 | 0.0911 |
| 2 | -0.0320 | 0.1705 | 0.1165 |
| 3 | -0.0402 | 0.1802 | 0.1182 |
| 4 | -0.0177 | 0.1921 | 0.1253 |
| 5 | 0.0166 | 0.2044 | 0.1196 |

**Best aggregate**: trend=**0.0166** (layer 5), seasonal=**0.2044** (layer 5), residual=**0.1253** (layer 4).

**Random backbone control** (M4 Hourly): trend=-0.4040, seasonal=0.1036, residual=-0.0827.

## Per-dataset tables

### M4 Hourly
N_use=414 (train=331, val=83), ctx_len=96, RS_THRESH exceeded: 0

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.1476 | 0.0469 | -0.0511 |
| 1 | -0.1812 | 0.1070 | 0.0003 |
| 2 | -0.1553 | 0.1440 | 0.0322 |
| 3 | -0.2221 | 0.1375 | 0.0102 |
| 4 | -0.1411 | 0.1818 | 0.0107 |
| 5 | -0.0054 | 0.2245 | 0.0085 |
| NULL | -0.0001 | -0.0000 | -0.0011 |

### M4 Monthly
N_use=1000 (train=800, val=200), ctx_len=36, RS_THRESH exceeded: 19

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.0350 | 0.3635 | 0.0020 |
| 1 | -0.0303 | 0.3879 | 0.0613 |
| 2 | -0.0230 | 0.4127 | 0.1115 |
| 3 | -0.0084 | 0.4011 | 0.1235 |
| 4 | -0.0001 | 0.3948 | 0.1454 |
| 5 | 0.0011 | 0.3950 | 0.1461 |
| NULL | -0.0094 | -0.0000 | -0.0002 |

### M4 Daily
N_use=2000 (train=1600, val=400), ctx_len=93, RS_THRESH exceeded: 44

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | 0.0433 | -0.2811 | -0.0499 |
| 1 | 0.0597 | -0.1143 | 0.0819 |
| 2 | 0.0666 | -0.0412 | 0.0878 |
| 3 | 0.0749 | 0.0195 | 0.1061 |
| 4 | 0.0865 | 0.0300 | 0.1135 |
| 5 | 0.0881 | 0.0260 | 0.0802 |
| NULL | -0.0053 | -0.0000 | -0.0000 |

### M4 Weekly
N_use=359 (train=287, val=72), ctx_len=80, RS_THRESH exceeded: 4

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.1730 | 0.2723 | 0.2515 |
| 1 | -0.0095 | 0.3545 | 0.2774 |
| 2 | 0.0248 | 0.3646 | 0.3064 |
| 3 | 0.0303 | 0.3729 | 0.3041 |
| 4 | 0.0463 | 0.3649 | 0.3090 |
| 5 | 0.0788 | 0.3788 | 0.3133 |
| NULL | -0.0002 | -0.0017 | -0.0000 |

### Electricity
N_use=370 (train=296, val=74), ctx_len=336, RS_THRESH exceeded: 0

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.0509 | -0.0373 | 0.0176 |
| 1 | -0.0885 | -0.0439 | 0.0348 |
| 2 | -0.0729 | -0.0277 | 0.0445 |
| 3 | -0.0757 | -0.0300 | 0.0473 |
| 4 | -0.0802 | -0.0110 | 0.0481 |
| 5 | -0.0797 | -0.0024 | 0.0499 |
| NULL | -0.0056 | -0.0000 | -0.0000 |

## Control: Random Backbone (M4 Hourly)

| Layer | Trend | Seasonal | Residual |
|-------|-------|----------|----------|
| 0 | -0.6514 | 0.1032 | -0.1195 |
| 1 | -0.4040 | 0.1036 | -0.0827 |
| 2 | -0.6132 | 0.1036 | -0.1132 |
| 3 | -0.9860 | 0.1036 | -0.2000 |
| 4 | -1.0742 | 0.1026 | -0.2168 |
| 5 | -0.9787 | 0.1020 | -0.1958 |

## Verdict: MIXED

MIXED — trend=POOR  seasonal=PARTIAL.
Trend R²=0.0166 (layer 5), Seasonal R²=0.2044 (layer 5).
Phase 2 scoped per-component: teach only the missing component(s),
reuse frozen backbone representations for those already present.
