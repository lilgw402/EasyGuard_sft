#! /usr/bin/env bash

# Usage:
# bash ./lint_py.sh --diff      # 列出修复前后的diff
# bash ./lint_py.sh --inplace   # 直接原地修复

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../

# hack for code format
if [ -f "${THIS_PATH}/pre-commit" ];then
    cp ${THIS_PATH}/pre-commit ${ROOT_PATH}/.git/hooks/pre-commit
fi

base="--cached"
target="origin/master"

fail_on_diff=0
has_diff=0
diff=0
inplace=0

while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [-h|--help] [-d|--diff] [-i|--inplace]"
            exit 0
        ;;
        -d|--diff)
            diff=1
            shift
        ;;
        -i|--inplace)
            inplace=1
            shift
        ;;
        *)
            echo "error: unrecognized option $1"
            exit 1
    esac
done

if ! autopep8 --version > /dev/null; then
    echo "error: no autopep8 found."
    exit 1
fi

files_to_check=$(git diff $base $target --name-only --diff-filter=ACMRT | grep "\.py$")
if [ "${files_to_check}" = "" ]; then
    exit 0
fi

opts="-d"
if [ "${inplace}" = "1" ]; then
    opts="-i"
fi

for f in ${files_to_check}; do
    result=$(autopep8 ${opts} --max-line-length=180 --ignore E402,E722,E731 --aggressive --aggressive --recursive ${f})
    if [ "$result" != "" ]; then
        echo "===== $f ====="
        echo -e "$result"
    fi
done

if [ "$fail_on_diff" -eq 1 ] && [ "$has_diff" -eq 1 ]; then
    exit 1
fi

