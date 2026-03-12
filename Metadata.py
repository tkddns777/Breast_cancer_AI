import pandas as pd
from sklearn.model_selection import train_test_split


def build_rsna_metadata(
    csv_path,
    save_path,
    train_ratio=0.7,
    val_ratio=0.15,
    random_state=42
):

    df = pd.read_csv(csv_path)

    df = df[df["view"].isin(["CC", "MLO"])]

    df["view_name"] = df["laterality"] + "_" + df["view"]

    patients = []

    for patient_id, g in df.groupby("patient_id"):

        views = set(g["view_name"])

        required = {"L_CC","L_MLO","R_CC","R_MLO"}

        if not required.issubset(views):
            continue

        row = {}

        row["patient_id"] = patient_id
        row["label"] = int(g["cancer"].max())

        view_map = {}

        for _, r in g.iterrows():

            key = r["laterality"] + "_" + r["view"]

            if key not in view_map:
                view_map[key] = r["image_id"]

        row["L_CC"] = view_map["L_CC"]
        row["L_MLO"] = view_map["L_MLO"]
        row["R_CC"] = view_map["R_CC"]
        row["R_MLO"] = view_map["R_MLO"]

        patients.append(row)

    patient_df = pd.DataFrame(patients)

    print("usable patients:", len(patient_df))

    # train / temp
    train_df, temp_df = train_test_split(
        patient_df,
        test_size=(1-train_ratio),
        stratify=patient_df["label"],
        random_state=random_state
    )

    # val / test
    val_size = val_ratio / (1-train_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1-val_size),
        stratify=temp_df["label"],
        random_state=random_state
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    meta_df = pd.concat([train_df,val_df,test_df],ignore_index=True)

    meta_df.to_csv(save_path,index=False)

    print("metadata saved:",save_path)

    print("\nSplit distribution")
    print(meta_df.groupby(["split","label"]).size())

    return meta_df

meta = build_rsna_metadata(
    csv_path=r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\train.csv",
    save_path=r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\metadata.csv"
)