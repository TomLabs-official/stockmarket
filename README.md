# Stock-market model site

This site serves the existing Python neural network through a Node.js HTTP server. The model trains once when the server starts, then stays running as a worker and answers prediction requests. No npm or Python packages beyond the Python standard library are required.

## Run locally

Install Node.js 20 or newer and Python 3. Then, from this directory, run:

```sh
npm start
```

Open `http://localhost:3000`. If Python is not available as `python3` on Linux/macOS or `python` on Windows, set `PYTHON_EXECUTABLE` to its executable path. The server listens on port `3000` by default; deployment platforms can supply `PORT`.

## Public deployment with Docker

Build and run the container:

```sh
docker build -t stockmarket-model .
docker run --rm -p 3000:3000 stockmarket-model
```

Deploy this directory as a Docker application on a container hosting platform, and configure its public port as `3000` (or use the `PORT` provided by the platform).

## Prediction API

`POST /api/predict` accepts JSON with eight numeric features in this exact order:

1. `return_yesterday`
2. `return_2_days_ago`
3. `return_3_days_ago`
4. `return_5_days_ago`
5. `return_10_days_ago`
6. `average_5_day_return`
7. `volatility_10_days`
8. `relative_volume`

Returns and volatility are decimal values (for example, `0.02` means 2%); relative volume is a ratio (for example, `1.2` means 20% above average). The page accepts percentages for the first seven values and converts them to decimals before sending them. Input bounds are ±5 for returns, 0–5 for volatility, and 0–1000 for relative volume.

Example:

```sh
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.02845,-0.02808,-0.01877,-0.03581,-0.03652,-0.0073,0.03154,1.2522]}'
```

The response contains the `Down`, `Neutral`, or `Up` guess and each output-node activation score. These scores are not calibrated probabilities.

## Model scope

The network is trained from the fixed examples in `nvidiadata.py`; this app does not fetch current market data or automatically calculate indicators. Predictions are experimental and for educational purposes only, not financial advice or a guarantee of future performance.
