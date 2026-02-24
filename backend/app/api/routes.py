from fastapi import APIRouter, UploadFile, File
import pandas as pd
from io import StringIO

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()

    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin1", errors="replace")

    df = None
    for sep in (";", ",", "\t"):
        try:
            df_try = pd.read_csv(StringIO(text), sep=sep, dtype=str)
            if df_try.shape[1] > 1:
                df = df_try
                break
        except Exception:
            continue

    if df is None:
        return {"error": "CSV konnte nicht eingelesen werden"}

    return {
        "columns": df.columns.tolist(),
        "rows": df.head(10).to_dict(orient="records")
    }
