# Architecture

```mermaid
flowchart LR
    CFG[Versioned YAML config] --> SYN[Deterministic synthetic generator]
    SYN --> SIG[Causal rolling features]
    SIG --> BT[Cost-aware backtest]
    BT --> MET[Metrics and drawdown]
    MET --> ART[CSV + JSON artifacts]
    NAV[Strategy NAV / holdings fixtures] --> ALLOC[Portfolio allocator]
    SCORES[Synthetic factor scores] --> ALLOC
    ALLOC --> SNAP[Auditable portfolio snapshot]
```

## Design principles

1. **No look-ahead:** rolling features are based on observations available before the current signal.
2. **Explicit costs:** position changes incur configurable transaction costs.
3. **Deterministic inputs:** fixed seeds make examples and tests reproducible.
4. **Small, inspectable outputs:** CSV and JSON artifacts are easy to reconcile.
5. **Disclosure by construction:** the public demo never needs private datasets.
