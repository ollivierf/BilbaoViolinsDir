
import nbformat

path = "c:/Users/froll/Documents/Labo/Projets/Violon/notebooks/Violon_3.1_CheckDirInvarianceWithSHOrientation.ipynb"
nb = nbformat.read(path, as_version=4)

for i, cell in enumerate(nb.cells):
    if "wigner_d_matrix_from_rotation_matrix" in cell.source:
        print(f"Found in cell index {i} (type: {cell.cell_type})")
        print("Source excerpt:")
        print(cell.source[:200])
