#!/bin/bash
set -e

IN=${1:-hello_world}

watchexec -w ~/noir-projects/$IN -e nr ./build-fast.sh $1
