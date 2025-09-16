from fastapi import FastAPI

app = FastAPI(title="XLA - Nhận diện biển báo giao thông")

@app.get("/")
def root():
    return {"message": "Backend FastAPI hoạt động!"}