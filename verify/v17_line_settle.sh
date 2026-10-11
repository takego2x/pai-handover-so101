#!/usr/bin/env bash
# V17 2 台の受け渡しで、動かした後に待たないと、下ろしきる前に指を閉じて掴み損ねる。
source "$(dirname "$0")/_lib.sh"
patch_copy spikes/line_two_arms.py $W/line_no_settle.py "n=100, settle=50):" "n=100, settle=0):"
printf "settle=0 : "; quiet python $W/line_no_settle.py | grep -o "LINE.*"
printf "settle=50: "; quiet python spikes/line_two_arms.py | grep -o "LINE.*"
