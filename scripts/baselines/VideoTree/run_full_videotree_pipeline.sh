#!/bin/bash
set -euo pipefail

SCRIPT_LABEL="VideoTree full pipeline"

notify_both() {
  local message="$1"
  notify.py "${message}" || true
  slack_notify.py "${message}" || true
}

on_exit() {
  local exit_code=$?
  if (( exit_code == 0 )); then
    notify_both "${SCRIPT_LABEL} completed"
  else
    notify_both "${SCRIPT_LABEL} failed (exit=${exit_code})"
  fi
}

trap on_exit EXIT

bash 1_2_generate_captions_from_vseek.sh
bash 1_prepare_videotree_data_from_vseek.sh
bash 2_run_videotree_pipeline.sh "lvb"
bash 2_run_videotree_pipeline.sh "lvbench"
bash 2_run_videotree_pipeline.sh "videomme"
bash 2_run_videotree_pipeline.sh "mlvu"
