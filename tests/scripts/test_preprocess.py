import pytest
import numpy as np
import re
import os
import pathlib
import pandas as pd
from tsa.dateconverter import DateConverter
from eda.preprocessing.eda_write_dataset import CoutureWriteEDADataset


test_datasets = [
    # ("Tips.csv", "total_bill"),
    # ("penguins.csv", "species"),
    # ("test_files/DailyDelhiClimateTimeTrain.csv", None),
    ("test_files/Electric_Production_time.csv", "IPG2211A2N"),
    # ("titanic.csv", "Survived")
]


@pytest.mark.parametrize("path,target", test_datasets)
class TestTimePreprocess:
    def test_timestamp_convert(self, path, target, cai_io_support):
        converter = DateConverter()
        df = cai_io_support.read_s3_file_as_pd(str(path))

        print("df head ", df.head())
        print("before convert ", df.dtypes)
        for col in df.columns:
            if str(df.dtypes[col]) == 'object':
                try:
                    df[col] = df[col].apply(lambda x: converter.convert_date(x, include_time=True, infer_format=True))
                    print(f"column time : {col}")
                except:
                    print(f"column string : {col}")

        print("after convert ", df.dtypes)
        print("df head ", df.head())


    def test_missing_values(self, path, target, cai_io_support):
        preprocess_method = ["skip", ""]
        couture_summary = CoutureWriteEDADataset(
            pathlib.Path(path),
            "",
            preprocess_method,
            target,
            cai_io_support)
        couture_summary.set_raw_dataset()

        print("before sort ", couture_summary.raw_data.head())
        couture_summary.raw_data.set_index('DATE', drop=True, append=False, inplace=True, verify_integrity=False)
        couture_summary.raw_data = couture_summary.raw_data.sort_index()
        print("after sort ", couture_summary.raw_data.head())

        print("NULL COUNT BEFORE: ", couture_summary.raw_data.isna().sum().sum())
        couture_summary.preprocess_missing_data()
        null_count = couture_summary.raw_data.isna().sum().sum()
        print("NULL COUNT AFTER: ", null_count)
        assert null_count == 0