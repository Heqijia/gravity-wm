#!/usr/bin/env bash
# Live view of the GPUs in your Slurm allocations, plus the queue.
#
# The GPUs live on compute nodes, so nvidia-smi has to be run there.  Calling
# `srun` once per refresh would create a job step every 10 seconds, so instead
# one lightweight collector is started per allocation; it loops nvidia-smi on
# the node and writes the result to a shared state directory, and this watcher
# only reads files.  The collector exits on its own once the watcher stops
# touching its heartbeat, so nothing is left running after you quit.
#
#   usage: tools/gpuwatch.sh [options]
#     -i, --interval SEC         GPU refresh, default 10
#     -q, --queue-interval SEC   queue refresh, default 3600
#     -u, --user NAME            whose jobs to watch, default $USER
#     -1, --once                 print one frame and exit
#         --ascii                ASCII bars instead of block characters
#         --no-collector         run nvidia-smi through srun each refresh
#                                instead of using a background collector
#     -h, --help
set -uo pipefail

INTERVAL=10
QUEUE_INTERVAL=3600
WATCH_USER=${USER}
ONCE=0
ASCII=${GPUWATCH_ASCII:-0}
USE_COLLECTOR=1

while [ $# -gt 0 ]; do
  case "$1" in
    -i|--interval) INTERVAL=$2; shift 2 ;;
    -q|--queue-interval) QUEUE_INTERVAL=$2; shift 2 ;;
    -u|--user) WATCH_USER=$2; shift 2 ;;
    -1|--once) ONCE=1; shift ;;
    --ascii) ASCII=1; shift ;;
    --no-collector) USE_COLLECTOR=0; shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for tool in squeue sinfo srun; do
  command -v "$tool" >/dev/null || { echo "$tool not found; is Slurm on PATH?" >&2; exit 1; }
done

STATE=${GPUWATCH_STATE:-$HOME/.cache/gpuwatch}
mkdir -p "$STATE"
HEARTBEAT=$STATE/heartbeat
WATCHER_NONCE=$$-$(date +%s)
COLLECTOR_PIDS=()

if [ $ASCII -eq 1 ]; then FULL='#'; EMPTY='.'; else FULL='█'; EMPTY='░'; fi
if [ -t 1 ]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'; C_GREY=$'\033[90m'
else
  C_RESET=; C_DIM=; C_BOLD=; C_RED=; C_GREEN=; C_YELLOW=; C_BLUE=; C_CYAN=; C_GREY=
fi

cleanup() {
  # Release the nodes this watcher claimed, so its collectors exit on their
  # next pass instead of waiting for the heartbeat to go stale.  The heartbeat
  # survives if another watcher still owns a node.
  local owned=0 owner
  for owner in "$STATE"/*.owner; do
    [ -f "$owner" ] || continue
    if [ "$(cat "$owner" 2>/dev/null)" = "$WATCHER_NONCE" ]; then rm -f "$owner"
    else owned=1; fi
  done
  [ "$owned" -eq 1 ] || rm -f "$HEARTBEAT"
  rm -f "$STATE"/*.spawn
  for pid in ${COLLECTOR_PIDS[@]+"${COLLECTOR_PIDS[@]}"}; do
    kill "$pid" 2>/dev/null
  done
  [ $ONCE -eq 1 ] || printf '\033[?25h\n'
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- rendering

bar() { # bar <fraction 0..100> <width> <colour>
  local pct=${1%.*} width=$2 colour=$3 filled i out=
  [ -z "$pct" ] && pct=0
  [ "$pct" -lt 0 ] 2>/dev/null && pct=0
  [ "$pct" -gt 100 ] 2>/dev/null && pct=100
  filled=$(( pct * width / 100 ))
  for ((i = 0; i < filled; i++)); do out+=$FULL; done
  for ((i = filled; i < width; i++)); do out+=$EMPTY; done
  printf '%s%s%s' "$colour" "$out" "$C_RESET"
}

pressure_colour() { # colour by how full something is
  local pct=${1%.*}
  [ -z "$pct" ] && pct=0
  if [ "$pct" -ge 85 ] 2>/dev/null; then printf '%s' "$C_RED"
  elif [ "$pct" -ge 60 ] 2>/dev/null; then printf '%s' "$C_YELLOW"
  else printf '%s' "$C_GREEN"; fi
}

load_colour() { # colour by how busy something is: idle is the thing to notice
  local pct=${1%.*}
  [ -z "$pct" ] && pct=0
  if [ "$pct" -ge 50 ] 2>/dev/null; then printf '%s' "$C_GREEN"
  elif [ "$pct" -ge 10 ] 2>/dev/null; then printf '%s' "$C_YELLOW"
  else printf '%s' "$C_RED"; fi
}

human_seconds() { # 93784 -> 1d 02h 03m
  local s=$1
  if [ "$s" -lt 0 ] 2>/dev/null; then printf 'now'; return; fi
  local d=$(( s / 86400 )) h=$(( (s % 86400) / 3600 )) m=$(( (s % 3600) / 60 ))
  if [ "$d" -gt 0 ]; then printf '%dd %02dh %02dm' "$d" "$h" "$m"
  elif [ "$h" -gt 0 ]; then printf '%dh %02dm' "$h" "$m"
  else printf '%dm' "$m"; fi
}

strip_ansi() { printf '%s' "$1" | sed $'s/\033\\[[0-9;]*m//g'; }

rule() { # rule <title> <width>; the title may contain colour codes, which do
         # not occupy columns, so measure the stripped text
  local title=$1 width=$2 plain dashes
  plain=$(strip_ansi "$title")
  dashes=$(( width - ${#plain} - 4 ))
  [ "$dashes" -lt 0 ] && dashes=0
  printf '%s── %s %s%s\n' "$C_GREY" "$title$C_GREY" "$(printf '─%.0s' $(seq 1 $dashes))" "$C_RESET"
}

# ---------------------------------------------------------------- slurm state

my_gpu_jobs() { # jobid<TAB>name<TAB>node<TAB>gpucount<TAB>seconds_left
  squeue -h -u "$WATCH_USER" -t RUNNING -o '%i|%j|%N|%b|%L' 2>/dev/null |
    while IFS='|' read -r jobid name node tres left; do
      case "$tres" in *gpu*) ;; *) continue ;; esac
      local count=${tres##*:}
      case "$count" in ''|*[!0-9]*) count=1 ;; esac
      local secs
      secs=$(awk -v t="$left" 'BEGIN{
        n=split(t,p,"-"); d=0; rest=t
        if (n==2) { d=p[1]; rest=p[2] }
        m=split(rest,q,":")
        if (m==3) s=q[1]*3600+q[2]*60+q[3]
        else if (m==2) s=q[1]*60+q[2]
        else s=q[1]
        print d*86400+s }')
      printf '%s\t%s\t%s\t%s\t%s\n' "$jobid" "$name" "$node" "$count" "$secs"
    done
}

# The collector body.  Runs on the compute node; writes two files per node and
# exits once the watcher's heartbeat goes stale, so quitting the watcher is
# enough to clean it up.
collector_body() {
  cat <<'COLLECTOR'
state=$1; node=$(hostname -s); beat=$2; period=$3; nonce=$4
# Claim this node.  A newer collector overwrites the owner file, and whoever no
# longer owns it exits on its next pass, so duplicates cannot accumulate even
# when several watchers are started.
printf '%s\n' "$nonce" > "$state/$node.owner"
while :; do
  [ -f "$beat" ] || exit 0
  age=$(( $(date +%s) - $(stat -c %Y "$beat" 2>/dev/null || echo 0) ))
  [ "$age" -gt 120 ] && exit 0
  [ "$(cat "$state/$node.owner" 2>/dev/null)" = "$nonce" ] || exit 0
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,uuid \
    --format=csv,noheader,nounits > "$state/$node.gpu.tmp" 2>/dev/null &&
    mv "$state/$node.gpu.tmp" "$state/$node.gpu"
  : > "$state/$node.proc.tmp"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
  while IFS=',' read -r uuid pid mem; do
    uuid=${uuid// /}; pid=${pid// /}; mem=${mem// /}
    [ -z "$pid" ] && continue
    label=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    script=$(printf '%s' "$label" | grep -oE '[^ /]+\.py' | head -1)
    tag=$(printf '%s' "$label" | grep -oE '\-\-output-dir +[^ ]+' | head -1 | awk '{print $2}')
    [ -z "$tag" ] && tag=$(printf '%s' "$label" | grep -oE '\-\-output +[^ ]+' | head -1 | awk '{print $2}')
    printf '%s|%s|%s|%s|%s\n' "$uuid" "$pid" "$mem" "${script:-?}" "$(basename "${tag:-}" 2>/dev/null)" \
      >> "$state/$node.proc.tmp"
  done
  mv "$state/$node.proc.tmp" "$state/$node.proc"
  sleep "$period"
done
COLLECTOR
}

ensure_collector() { # ensure_collector <jobid> <node> <gpucount>
  local jobid=$1 node=$2 count=$3 file=$STATE/$node.gpu lock=$STATE/$node.spawn age
  local grace=$(( INTERVAL * 3 + 20 ))
  if [ -f "$file" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$file" 2>/dev/null || echo 0) ))
    [ "$age" -lt "$grace" ] && return 0
  fi
  # Debounce.  A collector can take a few seconds to be scheduled, and without
  # this every frame would launch another one and the job steps would pile up.
  if [ -f "$lock" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || echo 0) ))
    [ "$age" -lt "$grace" ] && return 0
  fi
  touch "$lock" "$HEARTBEAT"
  srun --jobid="$jobid" --overlap -n1 --cpus-per-task=1 --gres=gpu:"$count" \
    bash -s "$STATE" "$HEARTBEAT" "$(( INTERVAL > 2 ? INTERVAL - 1 : 1 ))" "$WATCHER_NONCE" \
    <<<"$(collector_body)" >/dev/null 2>&1 &
  COLLECTOR_PIDS+=("$!")
}

snapshot_direct() { # fallback: one srun per refresh
  local jobid=$1 node=$2 count=$3
  srun --jobid="$jobid" --overlap -n1 --cpus-per-task=1 --gres=gpu:"$count" \
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,uuid \
    --format=csv,noheader,nounits 2>/dev/null > "$STATE/$node.gpu"
  : > "$STATE/$node.proc"
}

# ---------------------------------------------------------------- panels

render_gpus() {
  local width=$1 jobs any=0
  jobs=$(my_gpu_jobs)
  if [ -z "$jobs" ]; then
    rule "GPUs" "$width"
    printf '   %sno running job of %s holds a GPU%s\n' "$C_DIM" "$WATCH_USER" "$C_RESET"
    return
  fi
  while IFS=$'\t' read -r jobid name node count left; do
    [ -z "$jobid" ] && continue
    any=1
    if [ $USE_COLLECTOR -eq 1 ]; then ensure_collector "$jobid" "$node" "$count"
    else snapshot_direct "$jobid" "$node" "$count"; fi

    local file=$STATE/$node.gpu procs=$STATE/$node.proc
    rule "$C_BOLD$node$C_RESET  job $jobid  $C_CYAN$name$C_RESET  ${count}x GPU  ${C_DIM}walltime left $(human_seconds "$left")$C_RESET" "$width"
    if [ ! -s "$file" ]; then
      printf '   %sstarting collector on %s ...%s\n' "$C_DIM" "$node" "$C_RESET"
      continue
    fi
    local stale=$(( $(date +%s) - $(stat -c %Y "$file" 2>/dev/null || echo 0) ))
    [ "$stale" -gt $(( INTERVAL * 4 + 30 )) ] &&
      printf '   %sreading is %ss old%s\n' "$C_YELLOW" "$stale" "$C_RESET"

    while IFS=',' read -r idx gname util used total temp power plimit uuid; do
      idx=${idx// /}; util=${util// /}; used=${used// /}; total=${total// /}
      temp=${temp// /}; power=${power// /}; plimit=${plimit// /}; uuid=${uuid// /}
      gname=$(printf '%s' "$gname" | sed 's/^ *//; s/NVIDIA //')
      [ -z "$idx" ] && continue
      local mem_pct=0
      [ "${total:-0}" -gt 0 ] 2>/dev/null && mem_pct=$(( used * 100 / total ))
      local pw_pct=0
      pw_pct=$(awk -v a="${power:-0}" -v b="${plimit:-1}" 'BEGIN{printf "%d", (b>0)? a*100/b : 0}')

      printf '   %sGPU %s%s %-18s %sutil%s %s %s%3s%%%s   %stemp%s %s%s°C%s  %spower%s %4.0fW/%4.0fW\n' \
        "$C_BOLD" "$idx" "$C_RESET" "$gname" \
        "$C_DIM" "$C_RESET" "$(bar "$util" 24 "$(load_colour "$util")")" \
        "$(load_colour "$util")" "$util" "$C_RESET" \
        "$C_DIM" "$C_RESET" "$(pressure_colour "$(( temp * 100 / 90 ))")" "$temp" "$C_RESET" \
        "$C_DIM" "$C_RESET" "${power:-0}" "${plimit:-0}"
      printf '          %-18s %smem %s %s%5.1f/%5.1f GiB (%s%%)%s\n' "" \
        "$C_DIM$C_RESET" "$(bar "$mem_pct" 24 "$(pressure_colour "$mem_pct")")" \
        "$(pressure_colour "$mem_pct")" \
        "$(awk -v v="$used" 'BEGIN{printf "%.1f", v/1024}')" \
        "$(awk -v v="$total" 'BEGIN{printf "%.1f", v/1024}')" "$mem_pct" "$C_RESET"

      if [ -s "$procs" ]; then
        grep -F "$uuid" "$procs" 2>/dev/null | while IFS='|' read -r _u pid pmem script tag; do
          printf '          %s%-7s %-16s %6.1f GiB  %s%s\n' "$C_GREY" "$pid" \
            "${script:0:16}" "$(awk -v v="${pmem:-0}" 'BEGIN{printf "%.1f", v/1024}')" \
            "${tag:-}" "$C_RESET"
        done
      fi
    done < "$file"
  done <<< "$jobs"
  [ $any -eq 1 ] || true
}

render_queue() {
  local width=$1
  rule "queue  ${C_DIM}refreshed hourly$C_RESET" "$width"
  local pending
  pending=$(squeue -h -u "$WATCH_USER" -t PENDING,CONFIGURING -o '%i|%j|%T|%S|%r|%b|%L' 2>/dev/null)
  if [ -z "$pending" ]; then
    printf '   %snothing of %s is queued%s\n' "$C_DIM" "$WATCH_USER" "$C_RESET"
  else
    printf '   %s%-10s %-22s %-10s %-20s %-12s %s%s\n' "$C_DIM" \
      "JOBID" "NAME" "GPUS" "ESTIMATED START" "IN" "REASON" "$C_RESET"
    local now; now=$(date +%s)
    while IFS='|' read -r jobid name state start reason tres limit; do
      [ -z "$jobid" ] && continue
      local gpus=${tres##*gres/gpu}
      gpus=${gpus#:}
      [ -z "$gpus" ] && gpus='-'
      local when='unknown' delta='-'
      if [ -n "$start" ] && [ "$start" != "N/A" ]; then
        local epoch; epoch=$(date -d "$start" +%s 2>/dev/null)
        if [ -n "$epoch" ]; then
          when=$(date -d "$start" '+%Y-%m-%d %H:%M')
          delta=$(human_seconds $(( epoch - now )))
        fi
      fi
      printf '   %-10s %-22s %-10s %-20s %s%-12s%s %s\n' \
        "$jobid" "${name:0:22}" "$gpus" "$when" "$C_CYAN" "$delta" "$C_RESET" "$reason"
    done <<< "$pending"
  fi

  # Free a100s cluster-wide: Gres is what a node has, GresUsed what is taken.
  local free_line
  free_line=$(sinfo -h -p "${GPUWATCH_PARTITIONS:-quad,longgpu}" -N \
      -O 'partition:20,gres:60,gresused:60,statelong:20' 2>/dev/null |
    awk '{
      total=0; used=0
      if (match($0, /gpu:a100:[0-9]+\(IDX/)) {
        s = substr($0, RSTART, RLENGTH); sub(/\(IDX/, "", s); split(s, b, ":"); used = b[3]
      }
      if (match($0, /gpu:a100:[0-9]+\(S:/)) {
        s = substr($0, RSTART, RLENGTH); sub(/\(S:/, "", s); split(s, a, ":"); total = a[3]
      }
      T += total; U += used
    } END { if (T > 0) printf "%d of %d a100 free cluster-wide", T-U, T }')
  [ -n "$free_line" ] &&
    printf '   %s%s%s\n' "$C_DIM" "$free_line" "$C_RESET"
}

frame() {
  local width=${COLUMNS_NOW:-100} now
  now=$(date '+%Y-%m-%d %H:%M:%S')
  printf '%sgpuwatch%s  %s  %sgpu every %ss, queue every %ss%s\n\n' \
    "$C_BOLD$C_BLUE" "$C_RESET" "$now" "$C_DIM" "$INTERVAL" "$QUEUE_INTERVAL" "$C_RESET"
  render_gpus "$width"
  printf '\n'
  cat "$STATE/queue.panel" 2>/dev/null
  [ $ONCE -eq 1 ] || printf '\n%spress q or ctrl-c to quit%s\n' "$C_DIM" "$C_RESET"
}

refresh_queue_panel() {
  COLUMNS_NOW=$(tput cols 2>/dev/null || echo 100)
  render_queue "$COLUMNS_NOW" > "$STATE/queue.panel.tmp" 2>/dev/null
  mv "$STATE/queue.panel.tmp" "$STATE/queue.panel"
}

# ---------------------------------------------------------------- main loop

touch "$HEARTBEAT"
refresh_queue_panel
last_queue=$(date +%s)

if [ $ONCE -eq 1 ]; then
  COLUMNS_NOW=$(tput cols 2>/dev/null || echo 100)
  # A single snapshot cannot wait for a collector to start, so fall back to a
  # direct srun -- unless a collector from a running watcher has just written,
  # in which case reuse it and keep the per-process lines it provides.
  fresh=0
  for file in "$STATE"/*.gpu; do
    [ -f "$file" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$file") )) -lt $(( INTERVAL * 3 + 20 )) ] && fresh=1
  done
  [ $fresh -eq 1 ] || USE_COLLECTOR=0
  frame
  exit 0
fi

printf '\033[?25l'
while :; do
  COLUMNS_NOW=$(tput cols 2>/dev/null || echo 100)
  touch "$HEARTBEAT"
  now=$(date +%s)
  if [ $(( now - last_queue )) -ge "$QUEUE_INTERVAL" ]; then
    refresh_queue_panel
    last_queue=$now
  fi
  output=$(frame)
  printf '\033[H\033[2J%s' "$output"
  if [ -t 0 ]; then
    # On a terminal `read -t` doubles as the sleep and as the key handler.
    if read -rsn1 -t "$INTERVAL" key 2>/dev/null; then
      case "$key" in
        q|Q) break ;;
        r|R) refresh_queue_panel; last_queue=$(date +%s) ;;
      esac
    fi
  else
    # Without a terminal `read` would return at once and spin the loop.
    sleep "$INTERVAL"
  fi
done
