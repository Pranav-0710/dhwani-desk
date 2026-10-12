from fastapi import FastAPI

app = FastAPI(title="Dhwani Desk", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
