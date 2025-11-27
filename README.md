After cloning the repo, run

```Bash
uv sync
```

at the root of the repo.

---

To run a specific combination (e.g., `q_vs_pso`), either run,

```Bash
cd q_vs_pso
./run.sh
```

to run for all the shock conditions (none, scheme A, scheme B, scheme C),
or run,

```Bash
cd q_vs_pso
uv run q_vs_pso_schemeA.py
```

to simulate under a specific kind of shock configuration.
