#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# 配置区
########################################
GMX_BIN="${GMX_BIN:-gmx}"

STRUCTURE="1chian.tpr"
TRAJ="CG_centermol.xtc"
NDX="angle.ndx"

LOG_DIR="logs_gmx_angle"
mkdir -p "$LOG_DIR"

########################################
# 退出与报错处理
########################################
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Script failed with exit code $exit_code"
    else
        echo "[INFO] All gmx angle jobs finished successfully."
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
run_angle() {
    local group_name="$1"
    local out_file="$2"

    echo "--------------------------------------------------"
    echo "[INFO] Running group: $group_name"
    echo "[INFO] Output file : $out_file"
    echo "--------------------------------------------------"

    # 这里 group_name 必须和 angle.ndx 中的组名完全一致
    if ! printf '%s\n' "$group_name" | \
        "$GMX_BIN" angle \
            -n "$NDX" \
            -f "$TRAJ" \
            -type angle \
            -binwidth 1 \
            -od "$out_file" \
            > "$LOG_DIR/${out_file%.xvg}.log" 2>&1
    then
        echo "[ERROR] gmx angle failed for group: $group_name"
        echo "[ERROR] Check log: $LOG_DIR/${out_file%.xvg}.log"
        exit 1
    fi

    echo "[INFO] Finished: $out_file"
}

########################################
# 主程序
########################################
run_angle "RRR"      "angle_distRRR.xvg"
run_angle "RDE"      "angle_distRDE.xvg"
run_angle "DRRright" "angle_distDRRright.xvg"
run_angle "DRRleft"  "angle_distDRRleft.xvg"
