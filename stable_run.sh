#!/usr/bin/env bash

set -u

# Auto-resume wrapper for Ultralytics YOLO detect training.
# Usage example:
#   bash stable_run.sh \
#     model=rtdetr-l.pt \
#     data=ultralytics/cfg/datasets/VisDrone.yaml \
#     epochs=100 imgsz=640 batch=4 amp=False device=0 workers=8 \
#     project=/home/xjc/work/ultralytics/runs/detect \
#     name=rtdetr_train

MAX_RETRIES="${MAX_RETRIES:-100}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

if [[ $# -eq 0 ]]; then
	echo "Usage: bash stable_run.sh key=value [key=value ...]"
	exit 1
fi

TRAIN_ARGS=("$@")
PROJECT=""
NAME=""

has_arg_key() {
	local key="$1"
	local item
	for item in "${TRAIN_ARGS[@]}"; do
		[[ "$item" == "$key="* ]] && return 0
	done
	return 1
}

for arg in "${TRAIN_ARGS[@]}"; do
	case "$arg" in
		project=*) PROJECT="${arg#project=}" ;;
		name=*) NAME="${arg#name=}" ;;
	esac
done

if [[ -z "$PROJECT" || -z "$NAME" ]]; then
	echo "Error: both project=... and name=... are required."
	exit 2
fi

# Keep run directory stable and avoid Ultralytics auto-renaming (e.g., name -> name2).
if ! has_arg_key "exist_ok"; then
  TRAIN_ARGS+=("exist_ok=True")
fi

RUN_DIR="$PROJECT/$NAME"
WEIGHTS_DIR="$RUN_DIR/weights"
LAST_PT="$WEIGHTS_DIR/last.pt"
LOG_FILE="$RUN_DIR/auto_resume.log"
ARGS_FILE="$RUN_DIR/train_args.txt"

mkdir -p "$RUN_DIR"
printf "%s\n" "${TRAIN_ARGS[*]}" > "$ARGS_FILE"

# If a previous checkpoint exists, resume directly.
if [[ -f "$LAST_PT" ]]; then
  echo "[$(date '+%F %T')] Found existing checkpoint: $LAST_PT. Start with resume." | tee -a "$LOG_FILE"
  yolo detect train resume model="$LAST_PT" project="$PROJECT" name="$NAME"
  status=$?
else
  echo "[$(date '+%F %T')] Start train with args: ${TRAIN_ARGS[*]}" | tee -a "$LOG_FILE"
  yolo detect train "${TRAIN_ARGS[@]}"
  status=$?
fi

attempt=0
while [[ $status -ne 0 ]]; do
	if [[ $status -eq 130 ]]; then
		echo "[$(date '+%F %T')] Received interrupt (code=130). Stop without auto-resume." | tee -a "$LOG_FILE"
		exit "$status"
	fi

	if [[ ! -f "$LAST_PT" ]]; then
		echo "[$(date '+%F %T')] Train exited with code $status and $LAST_PT not found. Stop." | tee -a "$LOG_FILE"
		exit "$status"
	fi

	attempt=$((attempt + 1))
	if [[ $attempt -gt $MAX_RETRIES ]]; then
		echo "[$(date '+%F %T')] Exceeded MAX_RETRIES=$MAX_RETRIES. Stop." | tee -a "$LOG_FILE"
		exit "$status"
	fi

	echo "[$(date '+%F %T')] Train crashed (code=$status). Resume attempt $attempt/$MAX_RETRIES from $LAST_PT" | tee -a "$LOG_FILE"
	sleep "$SLEEP_SECONDS"
	yolo detect train resume model="$LAST_PT" project="$PROJECT" name="$NAME"
	status=$?
done

echo "[$(date '+%F %T')] Training finished successfully." | tee -a "$LOG_FILE"

