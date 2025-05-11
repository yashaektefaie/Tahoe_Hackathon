import anndata
import numpy as np

uce = anndata.read_h5ad("data/tahoe_universal_embeddings.h5ad")

control_condition = "[('DMSO_TF', 0.0, 'uM')]"

X_delta = np.zeros_like(uce.obsm["X_uce"])

for cell_line in uce.obs["cell_line"].unique():
    for plate in uce.obs["plate"].unique():
        cell_plate_mask = (uce.obs["cell_line"] == cell_line) & (uce.obs["plate"] == plate)
        control_mask = cell_plate_mask & (uce.obs["drugname_drugconc"] == control_condition)

        cell_plate_indices = np.where(cell_plate_mask)[0]
        control_indices = np.where(control_mask)[0]

        X_delta[cell_plate_indices] = uce.obsm["X_uce"][cell_plate_indices] - uce.obsm["X_uce"][control_indices]

        
uce.obsm["X_delta"] = X_delta

print("X_uce shape", uce.obsm["X_uce"].shape)
print("X_delta shape", uce.obsm["X_delta"].shape)

uce.write("data/tahoe_universal_embeddings_deltas.h5ad")