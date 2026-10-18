#!/usr/bin/env bash
# Build a ready-to-run cross-architecture checkout:  upstream repo @ pinned commit
# + our patch (if any) + our FiLM-Rate files overlaid.
#
#   bash ports/apply_ports.sh aegnn      [target_dir]    # default: ../external/aegnn
#   bash ports/apply_ports.sh eventmamba [target_dir]
#   bash ports/apply_ports.sh evtplus    [target_dir]
#
# Then export AEGNN_REPO / EVENTMAMBA_REPO / EVT_REPO to the target dir and use ports/<name>/slurm/*.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?usage: apply_ports.sh <aegnn|eventmamba|evtplus> [target_dir]}"
TARGET="${2:-$HERE/../external/$NAME}"

case "$NAME" in
  aegnn)      URL=https://github.com/uzh-rpg/aegnn.git;                    VAR=AEGNN_REPO ;;
  eventmamba) URL=https://github.com/rhwxmx/EventMamba.git;                VAR=EVENTMAMBA_REPO ;;
  evtplus)    URL=https://github.com/AlbertoSabater/EventTransformerPlus.git; VAR=EVT_REPO ;;
  *) echo "unknown port: $NAME" >&2; exit 1 ;;
esac
COMMIT="$(cat "$HERE/$NAME/patches/UPSTREAM_COMMIT.txt")"

[[ -d "$TARGET/.git" ]] || git clone "$URL" "$TARGET"
git -C "$TARGET" checkout -q "$COMMIT"
for p in "$HERE/$NAME"/patches/*.patch; do
  [[ -e "$p" ]] && git -C "$TARGET" apply --whitespace=nowarn "$p" && echo "applied $(basename "$p")"
done
# overlay our files (the overlay/ tree mirrors the upstream repo layout)
cp -r "$HERE/$NAME/overlay/." "$TARGET/"
echo "Ready: $TARGET   (upstream $URL @ ${COMMIT:0:10})"
echo "export $VAR=$(cd "$TARGET" && pwd)"
