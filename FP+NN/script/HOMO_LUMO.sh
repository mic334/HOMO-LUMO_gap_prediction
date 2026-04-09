#!/usr/bin/env bash

echo "idmol,homo,lumo"

for file in ../data/PUB_processed/xyz_files/xtb/*/dft/*log; do
    [ -f "$file" ] || continue
    
    idmol=$(basename "$(dirname "$(dirname "$file")")")

    awk -v id="$idmol" '
        /The electronic state is/ {
            homo = ""
            lumo = ""
            prev = ""
            inblock = 1
            next
        }

        inblock && /Alpha virt\. eigenvalues --/ && lumo == "" {
            # HOMO = ultimo campo della riga precedente
            n = split(prev, a, /[[:space:]]+/)
            for (i = n; i >= 1; i--) {
                if (a[i] != "") {
                    homo = a[i]
                    break
                }
            }

            # LUMO = primo numero dopo "--" nella riga corrente
            split($0, parts, /--/)
            gsub(/^[[:space:]]+/, "", parts[2])
            split(parts[2], b, /[[:space:]]+/)
            lumo = b[1]
        }

        {
            prev = $0
        }

        END {
            if (homo != "" && lumo != "")
                print id "," homo "," lumo
        }
    ' "$file"
done
