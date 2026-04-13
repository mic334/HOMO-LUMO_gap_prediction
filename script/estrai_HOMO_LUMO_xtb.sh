#!/usr/bin/env bash

echo "id,gap_ev"

#for file in ../data/PUB_processed/xyz_files/xtb/*/*.out; do
#per DFT veloci 
for file in ../data/PUB_processed/DFT_55_FP+NN/xyz_files/xtb/*/*.out; do
    [ -f "$file" ] || continue

    id=$(basename "$(dirname "$file")")
    gap=$(grep "HOMO-LUMO GAP" "$file" | sed -E 's/.*GAP[[:space:]]+([0-9.+-Ee]+)[[:space:]]+eV.*/\1/')

    if [ -n "$gap" ]; then
        echo "$id,$gap"
    fi
done