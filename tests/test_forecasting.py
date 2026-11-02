import pandas as pd
from app.validation import validate_frame
from models.preprocessing import future_exog_from_history

def test_future_exog_is_finite():
    d={"timestamp":pd.date_range("2024-01-01",periods=200,freq="h"),"series_id":"x","target":range(200)}
    for i in range(1,6): d[f"aux_{i}"]=1.; d[f"cov_{i}"]=2.
    d["cov_cat"]="a"
    df=validate_frame(pd.DataFrame(d))
    f=future_exog_from_history(df,pd.date_range("2024-01-10",periods=24,freq="h"))
    assert len(f)==24
    assert f.filter(regex="^(aux|cov)_").notna().all().all()
