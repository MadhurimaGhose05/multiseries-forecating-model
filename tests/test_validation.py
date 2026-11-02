import pandas as pd
import pytest
from app.validation import validate_frame, DataValidationError

def sample(n=60):
    d={"timestamp":pd.date_range("2024-01-01", periods=n, freq="h"),"series_id":"x","target":range(n)}
    for i in range(1,6): d[f"aux_{i}"]=0.; d[f"cov_{i}"]=1.
    d["cov_cat"]="a"
    return pd.DataFrame(d)

def test_valid(): assert len(validate_frame(sample())) == 60
def test_missing_column():
    with pytest.raises(DataValidationError): validate_frame(sample().drop(columns="cov_2"))
def test_duplicate():
    x=sample(); x.loc[1,"timestamp"]=x.loc[0,"timestamp"]
    with pytest.raises(DataValidationError): validate_frame(x)
