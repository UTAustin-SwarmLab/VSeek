#!/bin/bash
set -euo pipefail

bash 1_2_generate_captions_from_vseek.sh
bash 1_prepare_videotree_data_from_vseek.sh
bash 2_run_videotree_pipeline.sh "lvb"
bash 2_run_videotree_pipeline.sh "lvbench"
bash 2_run_videotree_pipeline.sh "videomme"
bash 2_run_videotree_pipeline.sh "mlvu"
