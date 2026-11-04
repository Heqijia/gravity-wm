#!/usr/bin/env bash
# Where the unattended suite has got to.
R=/fs/scratch/PAS3151/qijia387/gwm/runs/suite
W=/users/PAS3151/qijia387/Phys-WM/work
printf '== phase log ==\n'; tail -6 "$W/orchestrate.log" 2>/dev/null
printf '\n== running ==\n'
for p in $(pgrep -f "gravity-wm/scripts" 2>/dev/null); do
  [ "$p" = "$$" ] && continue
  cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
  s=$(printf '%s' "$cl" | grep -oE 'scripts/[a-z_]+\.py')
  o=$(printf '%s' "$cl" | grep -oE '\-\-output(-dir)? [^ ]+' | head -1 | sed 's|.*/||')
  v=$(printf '%s' "$cl" | grep -oE '\-\-variant [a-z]+' | awk '{print $2}')
  [ -n "$s" ] && printf '  %-22s %-16s %s\n' "$s" "${v:-}" "${o:-}"
done | sort -u
printf '\n== completed ==\n'
printf '  %-18s %8s %8s\n' model posttrain evals
for d in "$R"/*/; do
  n=$(basename "$d"); [ "$n" = report ] && continue
  printf '  %-18s %8s %8s\n' "$n" \
    "$(ls -d "$d"/post_*/posttrain_report.json 2>/dev/null | wc -l)" \
    "$(ls -d "$d"/post_*/evaluation.json 2>/dev/null | wc -l)"
done
printf '\n== failures ==\n'
grep -v 'exit=0' "$W/logs/_exits.txt" 2>/dev/null | tail -8 || echo '  none'
printf '\n== latest report ==\n'
ls -t "$R"/report/*.txt 2>/dev/null | head -1 | xargs -r head -10
