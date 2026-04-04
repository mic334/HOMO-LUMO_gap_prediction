import subprocess as sp
import os
import shutil 

class run_xtb:
    def __init__(self, path_inp):
        self.path = os.path.abspath(path_inp)
        self.xtb_dir = os.path.join(self.path, "xtb")
        os.makedirs(self.xtb_dir, exist_ok=True)

    def xtb(self, xyz_file):
        os.makedirs(self.xtb_dir, exist_ok=True)

        xyz_name = os.path.basename(xyz_file)
        base_name = os.path.splitext(xyz_name)[0]

        src = os.path.join(self.path, xyz_name)
        dst = os.path.join(self.xtb_dir, xyz_name)

        #evita di ricopiare se già esiste
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        out_file = os.path.join(self.xtb_dir, f"{base_name}.out")
        err_file = os.path.join(self.xtb_dir, f"{base_name}.err")

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
                cwd=self.xtb_dir,
                stdout=fout,
                stderr=ferr,
                text=True
            )

    