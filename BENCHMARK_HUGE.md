| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `./target/release/rg-cpu BENCHMARK_TARGET bench_huge.txt` | 134.8 ± 13.5 | 109.6 | 161.7 | 1.00 |
| `RG_GPU_THRESHOLD=2000000 ./target/release/rg-gpu BENCHMARK_TARGET bench_huge.txt` | 1491.8 ± 179.6 | 1232.3 | 1798.3 | 11.07 ± 1.73 |
