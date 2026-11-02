import pandas as pd
from app.validation import validate_frame
from models.preprocessing import make_feature_row

def test_unseen_series_id_does_not_enter_features():
    n=200
    d={"timestamp":pd.date_range("2024-01-01",periods=n,freq="h"),"series_id":"brand_new","target":range(n)}
    for i in range(1,6): d[f"aux_{i}"]=1.; d[f"cov_{i}"]=2.
    d["cov_cat"]="new_category"
    df=validate_frame(pd.DataFrame(d))
    row=make_feature_row(df.iloc[:-1],df.timestamp.iloc[-1],df.iloc[-1].to_dict())
    assert "series_id" not in row
