#!/bin/sh
# Chronological replication (train Jan-Apr, tune May, test June). Rules identical to the interleaved runs.
export PEMS_SPLIT=chrono
cd "$(dirname "$0")"
until [ -f results/day_params_d26_tau6s_chrono_test.pt ]; do sleep 30; done
.venv/bin/python src/replay.py > results/chrono_typical_log.txt 2>&1
.venv/bin/python src/estimate_2026.py results/wm_params_2026_typical_chrono.pt _typ 80 > results/nowcast_typ_chrono_log.txt 2>&1
.venv/bin/python src/run_replay.py tune > results/replay_tune_chrono_log.txt 2>&1
T=$(python3 -c "import json;print(json.load(open('results/replay_tune_summary_chrono.json'))['chosen_threshold'])")
.venv/bin/python src/run_replay.py test $T > results/replay_test_chrono_log.txt 2>&1
.venv/bin/python src/run_nas.py > results/nas_chrono_log.txt 2>&1
echo done > results/chrono_chain_done.txt
