#!/usr/bin/env bash

set -euo pipefail

uv run python scripts/analysis_error_structure.py
uv run python scripts/plot_error_structure.py

# uv run python scripts/analysis_free_energy.py \
#   --families SRAM_XMC4500 RO_Spartan7_roomT RO_Maiti_Spartan3E \
#   --devices-per-family 20 \
#   --group-sizes 1 2 4 6 8 10 12 16 20 \
#   --training-configs cd:1 cd:5 cd:10 pcd:1 \
#   --visible-bias-init marginal \
#   --gibbs-steps 1 10 50 100 \
#   --relaxation-steps 1 5 10 \
#   --target-aucs 0.8 0.9 0.95 0.99
# uv run python scripts/plot_free_energy.py
