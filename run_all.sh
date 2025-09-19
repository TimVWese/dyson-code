#!/bin/bash

julia --project -e 'using Pkg; Pkg.instantiate()'

# Find all Julia files in the script directory and run them
for file in scripts/*.jl; do
    if [ -f "$file" ]; then
        echo "Running $file..."
        julia "$file"
        echo "Finished $file"
        echo "------------------------"
    fi
done
