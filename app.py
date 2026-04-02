import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from model import predict_image, classes

app = FastAPI(title="Ingredient Classifier API")


@app.get("/")
def root():
    return {
        "message": "API is running",
        "num_classes": len(classes)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là ảnh.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image, top_k=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")