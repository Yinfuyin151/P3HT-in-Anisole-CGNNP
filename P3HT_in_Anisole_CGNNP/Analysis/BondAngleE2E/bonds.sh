#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# 配置区
########################################
GMX_BIN="${GMX_BIN:-gmx}"

STRUCTURE="1chian.tpr"
TRAJ="CG_centermol.xtc"
NDX="bonds.ndx"

LOG_DIR="logs_gmx_distance"
mkdir -p "$LOG_DIR"

########################################
# 退出与报错处理
########################################
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Script failed with exit code $exit_code"
    else
        echo "[INFO] All gmx distance jobs finished successfully."
    fi
}
trap cleanup EXIT

on_interrupt() {
    echo
    echo "[WARN] Interrupted by user (Ctrl+C). Exiting safely..."
    exit 130
}
trap on_interrupt INT TERM

########################################
# 基础检查
########################################
check_file() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
}

check_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] Command not found: $1"
        exit 1
    fi
}

check_cmd "$GMX_BIN"
check_file "$STRUCTURE"
check_file "$TRAJ"
check_file "$NDX"

########################################
# 运行函数
########################################
run_distance() {
    local group_name="$1"
    local out_file="$2"

    echo "--------------------------------------------------"
    echo "[INFO] Running group: $group_name"
    echo "[INFO] Output file : $out_file"
    echo "--------------------------------------------------"

    # 用 here-string 把交互选项喂给 gmx
    # 这里 group_name 必须和 bonds.ndx 中的组名完全一致
    if ! printf '%s\n' "$group_name" | \
        "$GMX_BIN" distance \
            -s "$STRUCTURE" \
            -f "$TRAJ" \
            -n "$NDX" \
            -oh "$out_file" \
            -len 0.5 \
            -binw 0.001 \
            -dt 1 \
            -tu fs \
            > "$LOG_DIR/${out_file%.xvg}.log" 2>&1
    then
        echo "[ERROR] gmx distance failed for group: $group_name"
        echo "[ERROR] Check log: $LOG_DIR/${out_file%.xvg}.log"
        exit 1
    fi

    echo "[INFO] Finished: $out_file"
}

########################################
# 主程序
########################################
run_distance "bonds_RR" "bondRR.xvg"
run_distance "bonds_DE" "bondDE.xvg"
run_distance "bonds_RD" "bondRD.xvg"
run_distance "E2E"      "bondE2E.xvg"
