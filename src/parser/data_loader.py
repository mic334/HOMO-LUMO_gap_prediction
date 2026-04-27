from pyclbr import Class
import pandas as pd
from importlib.resources import path
from pathlib import Path
import pandas as pd
from sympy import limit
from rdkit import Chem




class QM9Parser:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def parse_file(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        n_atoms = int(lines[0])
        props = lines[1].split()

        molecule_id = props[1]
        norm_dipole_moment = float(props[4])
        norm_static_polarizability = float(props[5])
        homo = float(props[7])
        lumo = float(props[8])
        gap = float(props[9])

        smiles_line_index = 2 + n_atoms + 1
        smiles = lines[smiles_line_index].split()[0]

        return {
            "molecule_id": molecule_id,
            "smiles": smiles,
            "norm_dipole_moment": norm_dipole_moment,
            "norm_static_polarizability": norm_static_polarizability,
            "homo": homo,
            "lumo": lumo,
            "gap": gap,
        }

    def parse_folder(self):
        files = sorted(self.folder_path.glob("*.xyz"))

        records = []
        errors = []

        for file in files:
            try:
                #richiamo pars.fiele !!!!!!!!!!!!!
                rec = self.parse_file(file)
                records.append(rec)
            except Exception as e:
                errors.append((file.name, str(e)))

        df = pd.DataFrame(records)

        print(f"File letti bene: {len(records)}")
        print(f"File con errore: {len(errors)}")

        if errors:
            print("Primi errori:")
            for name, err in errors[:5]:
                print(f"- {name}: {err}")

        return df

    def save_csv(self, df, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"CSV salvato in: {output_path}")
        
        
    def extract_sdf_data(self,sdf_file,limit=10000):
        
        suppl = Chem.SDMolSupplier(sdf_file)
        smiles = []

        for mol in suppl:
            if mol is None:
                continue
        
            smi = Chem.MolToSmiles(mol, canonical=True)
            smiles.append(smi)

            if len(smiles) == limit:
                break

        return pd.DataFrame({"smiles": smiles})
    
from pathlib import Path
from rdkit import Chem

from pathlib import Path
from rdkit import Chem


class  Alchemy_parser:
    def __init__(self):
        pass


    def sdf_file_to_id_smiles(self, sdf_path):
        """
        Input:
            sdf_path: path del file .sdf

        Output:
            (molecule_id, smiles)

        Se non riesce a leggere il file, restituisce:
            (molecule_id, None)
        """
        sdf_path = Path(sdf_path)
        molecule_id = sdf_path.stem

        try:
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
            if len(suppl) == 0:
                return molecule_id, None

            mol = suppl[0]
            if mol is None:
                return molecule_id, None

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return molecule_id, smiles

        except Exception:
            return molecule_id, None
        
    

    def in_join(self, first_data_frame, second_dataframe, col_id="gdb_idx", col_gap="gap\r\n(Ha, LUMO-HOMO)"):
        return first_data_frame.merge(
            second_dataframe[[col_id, col_gap]].rename(columns={col_gap: "gap"}),
            on=col_id,
            how="inner"
        )