#!/bin/bash
# Poll a Kaggle kernel until it stops, then download and summarise its output.
#
#   ./monitor.sh thorballer/rosa-definitive-match-length out
#
# Why this exists: `kaggle kernels output` silently returns nothing when the kernel is still
# running, which makes a live run look like an empty one. Always check status first.
set -u
ID="${1:?usage: monitor.sh <owner/kernel-slug> [out-dir]}"
OUT="${2:-out}"

for _ in $(seq 1 480); do
  st=$(kaggle kernels status "$ID" 2>/dev/null | grep -v outdated)
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "$(date +%H:%M) $st"; break ;;
  esac
  sleep 45
done

rm -rf "$OUT"
kaggle kernels output "$ID" -p "$OUT" >/dev/null 2>&1

# Results come back either as results.jsonl itself or wrapped in .log lines of the form
# {"data": "<json line>"}, depending on how the kernel wrote them. Handle both.
python3 - "$OUT" <<'PY'
import glob, json, os, sys

out = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(out, "*.log")) + glob.glob(os.path.join(out, "*.jsonl")):
    for line in open(f, errors="ignore"):
        line = line.strip().rstrip(",")
        try:
            obj = json.loads(line)
        except Exception:
            continue
        data = obj.get("data") if isinstance(obj, dict) else None
        if isinstance(data, str) and data.lstrip().startswith("{"):
            try:
                obj = json.loads(data)
            except Exception:
                continue
        rows.append(obj)

print(f"{len(rows)} records")
for r in rows:
    print(json.dumps(r)[:400])
PY
