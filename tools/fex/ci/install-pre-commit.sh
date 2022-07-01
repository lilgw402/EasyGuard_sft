#! /usr/bin/env bash

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
# hack for code format
if [ -f "${THIS_PATH}/pre-commit" ];then
    cp ${THIS_PATH}/pre-commit ${ROOT_PATH}/.git/hooks/pre-commit
fi

