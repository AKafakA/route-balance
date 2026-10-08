# RB CORE Pareto frontier — measured, 2026-07-31T07:45Z

Source: the 32 completed RouteBalance P1 cells (8 weight settings x 4 CORE rates),
corrected basis (minmax default, synced v3 tree, manifest d669faa2). Quality is the
DeepEval judge-of-record joined from the scored corpus; cost bills input+output at
3B/7B/14B/72B = $0.06/0.07/0.15/0.40 per 1M.

```

=== lambda 6 ===   (8/8 on the frontier)
  weights q/l/c     quality  cost$/1k  mean ms   pareto
  0.8/0.1/0.1        0.5121   0.04990     4216   *
  0.6/0.2/0.2        0.5048   0.03731     3325   *
  0.5/0.3/0.2        0.5014   0.03078     2813   *
  0.33/0.33/0.33     0.4909   0.02436     2369   *
  0.2/0.6/0.2        0.4903   0.02451     2222   *
  0.1/0.8/0.1        0.4870   0.02383     2137   *
  0.2/0.2/0.6        0.4868   0.02130     2333   *
  0.1/0.1/0.8        0.4808   0.01753     2276   *

=== lambda 12 ===   (8/8 on the frontier)
  weights q/l/c     quality  cost$/1k  mean ms   pareto
  0.8/0.1/0.1        0.5129   0.05023     4345   *
  0.6/0.2/0.2        0.5043   0.03813     3549   *
  0.5/0.3/0.2        0.5016   0.03099     2987   *
  0.33/0.33/0.33     0.4909   0.02409     2486   *
  0.2/0.6/0.2        0.4881   0.02394     2338   *
  0.1/0.8/0.1        0.4864   0.02333     2285   *
  0.2/0.2/0.6        0.4857   0.02118     2450   *
  0.1/0.1/0.8        0.4803   0.01754     2371   *

=== lambda 24 ===   (8/8 on the frontier)
  weights q/l/c     quality  cost$/1k  mean ms   pareto
  0.8/0.1/0.1        0.5133   0.05054     4639   *
  0.6/0.2/0.2        0.5065   0.03825     3709   *
  0.5/0.3/0.2        0.5016   0.03167     3231   *
  0.33/0.33/0.33     0.4918   0.02444     2691   *
  0.2/0.6/0.2        0.4882   0.02403     2655   *
  0.1/0.8/0.1        0.4868   0.02333     2461   *
  0.2/0.2/0.6        0.4866   0.02116     2634   *
  0.1/0.1/0.8        0.4815   0.01755     2534   *

=== lambda 30 ===   (7/8 on the frontier)
  weights q/l/c     quality  cost$/1k  mean ms   pareto
  0.8/0.1/0.1        0.5131   0.05026     5360   *
  0.6/0.2/0.2        0.5056   0.03829     4891   *
  0.5/0.3/0.2        0.5013   0.03165     4647   *
  0.33/0.33/0.33     0.4925   0.02447     3183   *
  0.1/0.8/0.1        0.4886   0.02339     2681   *
  0.2/0.6/0.2        0.4880   0.02415     2893   
  0.2/0.2/0.6        0.4858   0.02114     3661   *
  0.1/0.1/0.8        0.4818   0.01755     4745   *
```

## What this answers

**Q: can more weight points achieve Pareto-optimality at each rate?**
They are already there. 8/8 settings are non-dominated at lambda 6/12/24, and 7/8 at lambda 30
(0.2/0.6/0.2 is dominated by 0.1/0.8/0.1 on all three axes). Extra points therefore DENSIFY the
frontier; they do not repair non-optimality.

**The quality-cost frontier is rate-invariant.** 0.8/0.1/0.1 yields quality 0.5121/0.5129/0.5133/
0.5131 and cost $0.0499/0.0502/0.0505/0.0503 at lambda 6/12/24/30. Only latency moves
(4216 -> 5360 ms). This is measured support for the posted W4a claim that presets need no
retuning solely as rate changes — previously asserted without this evidence.

**Where the frontier is actually sparse: the high-quality end.** 0.6/0.2/0.2 -> 0.8/0.1/0.1 buys
+0.0086 quality for +32% cost, and nothing lies between w_quality 0.6 and 0.8. The low-quality end
is already dense (five settings between 0.4803 and 0.4925).

Revised densification proposal (supersedes the 2026-07-31T04:10Z set, which mostly added points to
the already-dense region):

    0.70:0.15:0.15   0.70:0.20:0.10   0.70:0.10:0.20
    0.45:0.35:0.20   0.40:0.30:0.30   0.55:0.25:0.20

**Rate for weight tuning: lambda 24.** Quality/cost are rate-invariant so a single rate transfers;
lambda 24 matches the basis of W1-A and W1-B, and a full 29-config sweep costs 1.9 h against 4.9 h
at lambda 6.
