import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_center_data(base_path):
    centers = ["Center1", "Center2", "Center3", "Center4"]

    for c in centers:
        path = os.path.join(base_path, c)
        excel_path = os.path.join(path, "h5_information.xlsx")

        if not os.path.exists(excel_path):
            print(f"{c} 找不到 h5_information.xlsx")
            continue


        df = pd.read_excel(excel_path)

        # 檢查 label 是否存在
        if "label" not in df.columns:
            print(f" {c} 沒有 label 欄位")
            continue

        # train / test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=57,
            stratify=df["label"] if df["label"].nunique() > 1 else None
        )

        # 存檔
        train_path = os.path.join(path, f"{c.lower()}_train.csv")
        test_path = os.path.join(path, f"{c.lower()}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"{c} 完成 | train: {len(train_df)} | test: {len(test_df)}")

if __name__ == "__main__":
    base_path = "/home/yuihsin/FedBCa/process_classification_v2"
    split_center_data(base_path)