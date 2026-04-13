import os
import shutil
import subprocess as sp

class run_xtb:
    def __init__(self, path_inp):
        self.path = os.path.abspath(path_inp)
        self.xtb_dir = os.path.join(self.path, "xtb")
        os.makedirs(self.xtb_dir, exist_ok=True)

    def xtb(self, xyz_file):
        xyz_name = os.path.basename(xyz_file)
        base_name = os.path.splitext(xyz_name)[0]

        # 📁 cartella specifica per molecola
        mol_dir = os.path.join(self.xtb_dir, base_name)
        os.makedirs(mol_dir, exist_ok=True)

        src = os.path.join(self.path, xyz_name)
        dst = os.path.join(mol_dir, xyz_name)

        # evita ricopia se esiste
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        out_file = os.path.join(mol_dir, f"{base_name}.out")
        err_file = os.path.join(mol_dir, f"{base_name}.err")

        # ambiente xtb
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OMP_STACKSIZE"] = "4G"

        with open(out_file, "w") as fout, open(err_file, "w") as ferr:
            sp.run(
                [
                    'xtb',
                    xyz_name,
                    '--opt',
                    '--gfn2',
                    '--chrg', '0',
                    '--cmaes',
                    '--maxstep', '0.2',
                    '--cycles', '1000',
                    '--gfniter', '3000',
                    '--namespace', base_name
                ],
                cwd=mol_dir,   # 🔥 cambia qui
                stdout=fout,
                stderr=ferr,
                text=True,
                env=env
            )