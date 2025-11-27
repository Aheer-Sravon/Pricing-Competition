#!/usr//bin/env bash

mkdir -p ./results

uv run ./pso_vs_ddpg.py
uv run ./pso_vs_ddpg_schemeA.py
uv run ./pso_vs_ddpg_schemeB.py
uv run ./pso_vs_ddpg_schemeC.py
