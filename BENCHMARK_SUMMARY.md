| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `./target/release/rg-cpu BENCHMARK_TARGET bench_large.txt` | 31.0 ± 1.6 | 28.8 | 36.0 | 1.00 |
| `RG_GPU_THRESHOLD=2000000 ./target/release/rg-gpu BENCHMARK_TARGET bench_large.txt` | 296.2 ± 13.6 | 277.7 | 320.8 | 9.55 ± 0.65 |
