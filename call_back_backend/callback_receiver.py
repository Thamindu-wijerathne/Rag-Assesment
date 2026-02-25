from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/callback")
async def callback(request: Request):
    data = await request.json()
    print("\nCALLBACK RECEIVED:\n", data)
    return {"ok": True}