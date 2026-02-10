import uvicorn
from src.pipeline import BTCPredictionPipeline


if __name__ == "__main__":
    # Option 1: Run prediction once
    pipeline = BTCPredictionPipeline()
    result = pipeline.run()
    print(result)

    # Option 2: Run as API server
    # uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)