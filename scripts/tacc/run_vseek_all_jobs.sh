#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/tacc/run_vseek_all_jobs.sh \
#       --algorithms "gspo rloo grpo" \
#       --cascade-deps 2 \
#       --reward-type em
#
# Meaning:
# - --cascade-deps N creates N afterany retries per algorithm.
# - Total jobs per algorithm = 1 initial + N dependent retries.

ALGORITHMS=("grpo")
CASCADE_DEPS=0
REWARD_TYPE="em"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithms)
            read -r -a ALGORITHMS <<< "$2"
            shift 2
            ;;
        --cascade-deps)
            CASCADE_DEPS="$2"
            shift 2
            ;;
        --reward-type)
            REWARD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --algorithms \"gspo rloo grpo\" --cascade-deps N --reward-type {em|puls}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$CASCADE_DEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: --cascade-deps must be a non-negative integer."
    exit 1
fi

if [[ "$REWARD_TYPE" != "em" && "$REWARD_TYPE" != "puls" ]]; then
    echo "Error: --reward-type must be 'em' or 'puls'."
    exit 1
fi

declare -A SCRIPT_MAP=(
    ["grpo"]="scripts/tacc/run_vseek_all_jobs.slurm"
    ["gspo"]="scripts/tacc/run_vseek_all_jobs_gspo.slurm"
    ["rloo"]="scripts/tacc/run_vseek_all_jobs_rloo.slurm"
)

# Vista may print a banner around sbatch output. Extract a clean job ID.
submit_and_get_jid() {
    local output jid
    output="$(sbatch --parsable "$@")" || {
        echo "sbatch submission failed." >&2
        echo "$output" >&2
        return 1
    }

    # Accept common sbatch ID formats:
    # - 123456
    # - 123456;cluster
    # - 123456_7 (array task)
    jid="$(printf "%s\n" "$output" | awk '/^[0-9]+([;_].*)?$/ {last=$1} END {print last}')"

    if [[ -z "$jid" ]]; then
        echo "Could not parse job id from sbatch output:" >&2
        echo "$output" >&2
        return 1
    fi

    echo "$jid"
}

echo "Algorithms     : ${ALGORITHMS[*]}"
echo "Cascade deps   : ${CASCADE_DEPS}"
echo "Reward type    : ${REWARD_TYPE}"
echo

for raw_alg in "${ALGORITHMS[@]}"; do
    alg="$(echo "$raw_alg" | tr '[:upper:]' '[:lower:]')"
    slurm_script="${SCRIPT_MAP[$alg]:-}"

    if [[ -z "$slurm_script" ]]; then
        echo "Error: Unsupported algorithm '$raw_alg'. Supported: gspo, rloo, grpo."
        exit 1
    fi

    if [[ ! -f "$slurm_script" ]]; then
        echo "Error: Slurm script not found: $slurm_script"
        exit 1
    fi

    echo "Submitting chain for algorithm: $alg"

    first_jid="$(submit_and_get_jid "$slurm_script" "$REWARD_TYPE")"
    echo "  attempt 1 (initial): $first_jid"
    prev_jid="$first_jid"

    for ((attempt=1; attempt<=CASCADE_DEPS; attempt++)); do
        next_try=$((attempt + 1))
        jid="$(submit_and_get_jid --dependency="afterany:${prev_jid}" "$slurm_script" "$REWARD_TYPE")"
        echo "  attempt ${next_try} (afterany:${prev_jid}): $jid"
        prev_jid="$jid"
    done

    echo
done

echo "All requested chains submitted."
