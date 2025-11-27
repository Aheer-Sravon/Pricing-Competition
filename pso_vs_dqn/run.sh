#!/usr/bin/env bash

mkdir -p ./results

uv run ./pso_vs_dqn.py
uv run ./pso_vs_dqn_schemeA.py
uv run ./pso_vs_dqn_schemeB.py
uv run ./pso_vs_dqn_schemeC.py
