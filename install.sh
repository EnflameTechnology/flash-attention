#!/bin/bash

dirs=(
    "build/"
    "dist/"
    "flash_attn.egg-info/"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done

python setup.py bdist_wheel

#pip uninstall flash_attn
pip install dist/*.whl --force-reinstall --no-deps
