!/bin/bash

# Scorre tutte le sottocartelle della directory corrente
for dir in */dft ; do
    echo ">>> Entro in $dir"
    cd "$dir" || continue

    # Lancia sbatch su tutti i file .sh (se esistono)
    if ls *.slurm 1> /dev/null 2>&1; then
        sbatch *.slurm
    else
        echo "Nessun file .slurm in $dir"
    fi

    cd ../..
done