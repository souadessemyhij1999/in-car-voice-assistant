| Backend | Task success | Unsafe actions executed | Unsafe requests refused | Handled in the car | Median latency | Cost per 1,000 commands |
|---|---|---|---|---|---|---|
| Offline in-car model | 81.4% | 0 | 8/8 | 87/118 | fallback 4 ms / local 7 ms | $0.00 |
| Claude only (cloud) | 95.8% | 0 | 8/8 | 0/118 | cloud 1811 ms | $0.52 |
| Hybrid: offline + Claude | 94.9% | 0 | 8/8 | 87/118 | cloud 1971 ms / local 4 ms | $0.13 |
