from fastapi import FastAPI

app = FastAPI(title="Dragon Boat Analysis API")


@app.get("/health")
def health():
    return {"status": "ok"}
