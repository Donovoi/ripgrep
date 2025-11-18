# DirectStorage/GDeflate Integration Benchmark Report

## Test Configuration

- **Date**: $(date)
- **System**: $(uname -a)
- **Ripgrep Version**: $(rg --version | head -1)

## Methodology

This benchmark compares search performance across different compression formats:
- **Uncompressed**: Direct file reading (baseline)
- **Gzip**: Traditional external process decompression (current ripgrep)
- **GDeflate**: Proposed native parallel decompression (future ripgrep)

### Test Scenarios

1. **Small files** (10KB - 100KB): Tests process overhead
2. **Medium files** (1MB - 10MB): Tests decompression throughput
3. **Large files** (10MB - 50MB): Tests parallel decompression scaling

Each test was run 3 times and averaged to reduce noise.

## Results

### Raw Data

```
File,Format,Size_Bytes,Avg_Time_MS,Runs,Status
code_large.txt,uncompressed,25700,4,3,success
code_large.txt,gzip,348,4,3,success
large_1.txt,uncompressed,39661930,13,3,success
large_1.txt,gzip,30140639,6,3,success
large_2.txt,uncompressed,33995938,10,3,success
large_2.txt,gzip,25834814,3,3,success
large_3.txt,uncompressed,25496954,10,3,success
large_3.txt,gzip,19376182,7,3,success
medium_1.txt,uncompressed,4249493,4,3,success
medium_1.txt,gzip,3229366,4,3,success
medium_2.txt,uncompressed,7082489,5,3,success
medium_2.txt,gzip,5382336,4,3,success
medium_3.txt,uncompressed,8498985,5,3,success
medium_3.txt,gzip,6458801,4,3,success
medium_4.txt,uncompressed,8498985,6,3,success
medium_4.txt,gzip,6458743,4,3,success
medium_5.txt,uncompressed,5665993,5,3,success
medium_5.txt,gzip,4305817,4,3,success
small_1.txt,uncompressed,106516,3,3,success
small_1.txt,gzip,80984,4,3,success
small_10.txt,uncompressed,78848,4,3,success
small_10.txt,gzip,59963,4,3,success
small_2.txt,uncompressed,112048,4,3,success
small_2.txt,gzip,85189,3,3,success
small_3.txt,uncompressed,127265,5,3,success
small_3.txt,gzip,96736,4,3,success
small_4.txt,uncompressed,110666,4,3,success
small_4.txt,gzip,84134,3,3,success
small_5.txt,uncompressed,132797,5,3,success
small_5.txt,gzip,100937,3,3,success
small_6.txt,uncompressed,127265,4,3,success
small_6.txt,gzip,96743,3,3,success
small_7.txt,uncompressed,62249,3,3,success
small_7.txt,gzip,47320,3,3,success
small_8.txt,uncompressed,125883,4,3,success
small_8.txt,gzip,95689,3,3,success
small_9.txt,uncompressed,80234,3,3,success
small_9.txt,gzip,61009,3,3,success
source_code.txt,uncompressed,257,3,3,success
source_code.txt,gzip,206,3,3,success
```

### Analysis

[1;33mAnalyzing results...[0m

[0;34mPerformance Summary:[0m
[0;34m════════════════════[0m

gzip           :     3.80 ms average,  1279.88 MB/s throughput
uncompressed   :     5.20 ms average,  1230.98 MB/s throughput

[0;34mSpeedup Factors (compared to uncompressed):[0m
[0;34m═══════════════════════════════════════════[0m

code_large.txt                 gzip      :  1.00x slower (-  0.0%)
large_1.txt                    gzip      :  0.46x slower (- 53.8%)
large_2.txt                    gzip      :  0.30x slower (- 70.0%)
large_3.txt                    gzip      :  0.70x slower (- 30.0%)
medium_1.txt                   gzip      :  1.00x slower (-  0.0%)
medium_2.txt                   gzip      :  0.80x slower (- 20.0%)
medium_3.txt                   gzip      :  0.80x slower (- 20.0%)
medium_4.txt                   gzip      :  0.67x slower (- 33.3%)
medium_5.txt                   gzip      :  0.80x slower (- 20.0%)
small_1.txt                    gzip      :  1.33x slower (--33.3%)
small_10.txt                   gzip      :  1.00x slower (-  0.0%)
small_2.txt                    gzip      :  0.75x slower (- 25.0%)
small_3.txt                    gzip      :  0.80x slower (- 20.0%)
small_4.txt                    gzip      :  0.75x slower (- 25.0%)
small_5.txt                    gzip      :  0.60x slower (- 40.0%)
small_6.txt                    gzip      :  0.75x slower (- 25.0%)
small_7.txt                    gzip      :  1.00x slower (-  0.0%)
small_8.txt                    gzip      :  0.75x slower (- 25.0%)
small_9.txt                    gzip      :  1.00x slower (-  0.0%)
source_code.txt                gzip      :  1.00x slower (-  0.0%)

