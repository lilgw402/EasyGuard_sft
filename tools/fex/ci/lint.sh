#!/usr/bin/env bash
set -e

python3 -m pylint fex tasks
python3 -m mypy --ignore-missing-imports fex tasks
