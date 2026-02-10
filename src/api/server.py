from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import schedule
import threading
import time

from src.pipeline import BTCPredictionPipeline

app = FastAPI(title="BTC Price Predictor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
pipeline = BTCPredictionPipeline()
latest_prediction = {}


def run_prediction():
    global latest_prediction
    latest_prediction = pipeline.run()


@app.on_event("startup")
async def startup():
    """Run initial prediction and schedule retraining."""
    run_prediction()
    schedule.every(24).hours.do(run_prediction)

    def scheduler_loop():
        while True:
            schedule.run_pending()
            time.sleep(60)

    threading.Thread(target=scheduler_loop, daemon=True).start()


@app.get("/predict")
async def get_prediction():
    return latest_prediction


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_prediction)
    return {"status": "Retraining started in background"}


@app.get("/health")
async def health():
    return {"status": "ok", "has_prediction": bool(latest_prediction)}