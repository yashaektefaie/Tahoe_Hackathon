import anndata
import pandas as pd

uce = anndata.read_h5ad("data/tahoe_universal_embeddings_deltas.h5ad")
vision = anndata.read_h5ad("data/tahoe_vision_scores.h5ad")

uce.obs = uce.obs.reset_index().rename(columns = {"index": "condition"})
vision.obs["condition"] = vision.obs.apply(
    lambda row: f"{row['Cell_ID_Cellosaur']}_{[(row['drug'], row['concentration'], row['concentration_unit'])]}_plate{row['plate']}", axis = 1
)

unique_cell_lines_uce = uce.obs["cell_line"].unique().tolist()
unique_drugs_uce = uce.obs["drugname_drugconc"].apply(
    lambda x: eval(x)[0][0]
).unique().tolist()

# print("number of unique cell lines:", len(unique_cell_lines_uce))
# print(unique_cell_lines_uce)

# print("\nnumber of unique drugs:", len(unique_drugs_uce))
# print(unique_drugs_uce)

conditions_uce = set(uce.obs["condition"].unique())
conditions_vision = set(vision.obs["condition"].unique())

only_in_uce = conditions_uce - conditions_vision
only_in_vision = conditions_vision - conditions_uce

with open("conditions_only_in_uce.txt", "w") as f:
    for condition in only_in_uce:
        f.write(f"{condition}\n")

with open("conditions_only_in_vision.txt", "w") as f:
    for condition in only_in_vision:
        f.write(f"{condition}\n")

vision = vision[vision.obs["condition"].drop_duplicates(keep = "first").index, :]
vision.obs = vision.obs.reset_index(drop = True)

merged_obs = pd.merge(
    uce.obs,
    vision.obs,
    on = "condition",
    how = "inner"
)

indices_in_vision = vision.obs.index[
    vision.obs["condition"].isin(merged_obs["condition"])
].tolist()

indices_in_uce = uce.obs.index[
    uce.obs["condition"].isin(merged_obs["condition"])
].tolist()

indices_in_vision = [int(x) for x in indices_in_vision]
indices_in_uce = [int(x) for x in indices_in_uce]

anndata_merged = anndata.AnnData(
    X = vision.X[indices_in_vision, :],
    obs = merged_obs,
    var = vision.var,
    obsm = {"X_uce" : uce.obsm["X_uce"][indices_in_uce, :],
            "X_delta": uce.obsm["X_delta"][indices_in_uce, :]}
)

anndata_merged.write("data/tahoe_vision_universal_embeddings.h5ad")