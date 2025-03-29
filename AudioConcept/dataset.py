from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from AudioConcept.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "features_30_sec.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
):
    logger.info(f"Processing dataset from {input_path}...")
    df = pd.read_csv(input_path, sep=";")
    
    df_filtered = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    logger.info("Calculating correlation matrix...")
    corr_matrix = df_filtered.corr()

    logger.info("Removing highly correlated features...")
    correlated_features = set()

    for i in tqdm(range(len(corr_matrix.columns)), desc="Identifying correlated features"):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname = corr_matrix.columns[i]
                if colname not in correlated_features:
                    if np.var(df_filtered[colname]) < np.var(df_filtered[corr_matrix.columns[j]]):
                        correlated_features.add(corr_matrix.columns[i])
                    else:
                        correlated_features.add(corr_matrix.columns[j])

    df = df.drop(columns=correlated_features)
    df = df.drop(columns={'filename', 'label'})

    logger.info(f"Number of features (columns) in final dataset: {df.shape[1]}")

    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df)

    logger.info("Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_standardized_df = pd.DataFrame(X_standardized, columns=df.columns)

    df_final = pd.concat([X_standardized_df, pd.Series(y_encoded, name="Y")], axis=1)

    logger.info(f"Number of features (columns) in final dataset: {df_final.shape[1]}")

    logger.info(f"Saving processed dataset to {output_path}...")
    df_final.to_csv(output_path, index=False)

    logger.success(f"Processing complete. Saved to {output_path}.")

if __name__ == "__main__":
    app()
