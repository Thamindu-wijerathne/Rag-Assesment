from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/callback")
async def callback(request: Request):
    data = await request.json()
    print("\nCALLBACK RECEIVED:\n", data)
    return {"ok": True}