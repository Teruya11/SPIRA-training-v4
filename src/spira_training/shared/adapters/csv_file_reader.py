import pandas as pd
from typing import List

from src.spira_training.shared.ports.file_reader import FileReader


class CSVFileReader(FileReader):
    def read(self, path: str) -> List[str]:
        df = pd.read_csv(path, header=0)
        self.validate_dataframe(df)
        return df.iloc[:, 0].str.strip().tolist()

    # We are assuming that the first column of the CSV contains the data
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty or df.iloc[:, 0].isnull().any() or (df.iloc[:, 0].astype(str).str.strip() == '').any():
            raise ValueError("A primeira coluna do CSV contém valores nulos ou vazios.")