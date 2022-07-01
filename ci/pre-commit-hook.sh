#! /usr/bin/env bash
set -x

function run_autopep8_diff() {
    local py_file=$1
    autopep8 \
    --diff \
    --max-line-length=180 \
    --ignore E402,E722,E731 \
    --aggressive --aggressive ${py_file}
}

base='--cached'
target='origin/master'
show_only=0
fail_on_diff=0
has_diff=0

while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [--fail-on-diff] [--show-only] [--base|-b <base(=--cached)>] [--target|-t <target(=origin/master)>]"
            exit 0
        ;;
        --fail-on-diff)
            fail_on_diff=1
            shift
        ;;
        --show-only)
            show_only=1
            shift
        ;;
        --base|-b)
            base="$2"
            shift
            shift
        ;;
        --target|-t)
            target="$2"
            shift
            shift
        ;;
        *)
            echo "error: unrecognized option $1"
            exit 1
    esac
done


files_to_check=`git diff $base $target --name-only --diff-filter=ACMRT | grep "\.py$"`
if [ "$files_to_check" = "" ]; then
    exit 0
fi

if ! autopep8 --version >/dev/null; then
    echo "error: autopep8 not found"
    exit 1
fi

if [ "$show_only" -eq 1 ]; then
    for f in $files_to_check; do
        result=$( run_autopep8_diff $f )
        if [ "$result" != "" ]; then
            echo "===== $f ====="
            echo -e "$result"
            has_diff=1
        fi
    done
fi

if [ "$fail_on_diff" -eq 1 ] && [ "$has_diff" -eq 1 ]; then
    exit 1
fi
