#!/bin/bash

# Iterate over command-line arguments
args=($@)

even_numbers=()

for ((i=0; i<${#args[@]}; i++)); do
    if [[ $((i % 2)) -eq 1 ]]; then
        even_numbers+=("${args[i]}")
    fi
done

echo "Even numbers from arguments: ${even_numbers[*]}"
echo "started sleep"
sleep 20
