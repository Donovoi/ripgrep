| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `target/release/rg -a "ABCDE" bench_huge.txt` | 768.5 ± 62.5 | 717.4 | 917.8 | 1.00 |
| `target/release/rg -a --gpu "ABCDE" bench_huge.txt` | 780.0 ± 61.0 | 709.3 | 874.9 | 1.02 ± 0.11 |
| `target/release/rg -a "A.B.C.D.E" bench_huge.txt` | 1299.2 ± 57.5 | 1247.7 | 1418.6 | 1.69 ± 0.16 |
| `target/release/rg -a --gpu "A.B.C.D.E" bench_huge.txt` | 1305.8 ± 48.9 | 1260.2 | 1398.6 | 1.70 ± 0.15 |
