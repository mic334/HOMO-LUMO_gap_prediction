#!/usr/bin/env bash

echo "idmol,homo,lumo,status"

for file in ../data/PUB_processed/xyz_files/xtb/*/dft/*.log; do
    [ -f "$file" ] || continue

    idmol=$(basename "$(dirname "$(dirname "$file")")")

    if tail -n 1 "$file" | grep -q "Normal termination"; then
        status="complete"
    else
        status="incomplete"
    fi

    awk -v id="$idmol" -v status="$status" '
        /The electronic state is/ {
            homo = ""
            lumo = ""
            prev = ""
            inblock = 1
            next
        }

        inblock && /Alpha virt\. eigenvalues --/ && lumo == "" {
            n = split(prev, a, /[[:space:]]+/)
            for (i = n; i >= 1; i--) {
                if (a[i] != "") {
                    homo = a[i]
                    break
                }
            }

            split($0, parts, /--/)
            gsub(/^[[:space:]]+/, "", parts[2])
            split(parts[2], b, /[[:space:]]+/)
            lumo = b[1]
        }

        { prev = $0 }

        END {
            if (homo != "" && lumo != "")
                print id "," homo "," lumo "," status
        }
    ' "$file"
done
