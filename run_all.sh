#!/bin/bash

julia +1.11.7 --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Find all Julia files in the script directory and run them
for file in scripts/*.jl; do
    if [ -f "$file" ]; then
        echo "Running $file..."
        julia +1.11.7 "$file"
        echo "Finished $file"
        echo "------------------------"
    fi
done
