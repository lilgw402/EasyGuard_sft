#!/usr/bin/env bash

set -e

echo "====================begin to test===================="
pip3 install pytest
pytest tests/unit_tests
echo "====================tests passed======================"