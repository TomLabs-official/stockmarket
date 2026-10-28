import json
import math
import random
import sys
from typing import Sequence

from nvidiadata import dataset


INPUT_SIZE = 8
HIDDEN_SIZE = 9
OUTPUT_SIZE = 3
EPOCHS = 10_000
CLASS_NAMES = ("Down", "Neutral", "Up")

_random = random.Random(42)
hidden_weights = [
    [_random.uniform(-1, 1) for _ in range(INPUT_SIZE)]
    for _ in range(HIDDEN_SIZE)
]
hidden_bias = [_random.uniform(-1, 1) for _ in range(HIDDEN_SIZE)]
output_weights = [
    [_random.uniform(-1, 1) for _ in range(HIDDEN_SIZE)]
    for _ in range(OUTPUT_SIZE)
]
output_bias = [_random.uniform(-1, 1) for _ in range(OUTPUT_SIZE)]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def sigmoid_derivative(output: float) -> float:
    return output * (1 - output)


def forward(inputs: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(inputs) != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} input features.")

    hidden_outputs = [
        sigmoid(
            sum(inputs[k] * hidden_weights[i][k] for k in range(INPUT_SIZE))
            + hidden_bias[i]
        )
        for i in range(HIDDEN_SIZE)
    ]
    outputs = [
        sigmoid(
            sum(
                hidden_outputs[i] * output_weights[j][i]
                for i in range(HIDDEN_SIZE)
            )
            + output_bias[j]
        )
        for j in range(OUTPUT_SIZE)
    ]
    return outputs, hidden_outputs


def get_accuracy(training_data: Sequence[tuple]) -> float:
    correct_predictions = 0

    for inputs, down, neutral, up in training_data:
        outputs, _ = forward(inputs)
        predicted_class = outputs.index(max(outputs))
        actual_class = (down, neutral, up).index(max(down, neutral, up))
        if predicted_class == actual_class:
            correct_predictions += 1

    return correct_predictions / len(training_data) * 100


def train_epoch() -> None:
    learning_rate = 0.765

    for inputs, target_down, target_neutral, target_up in dataset:
        outputs, hidden_outputs = forward(inputs)
        targets = (target_down, target_neutral, target_up)
        output_deltas = [
            (targets[j] - outputs[j]) * sigmoid_derivative(outputs[j])
            for j in range(OUTPUT_SIZE)
        ]
        hidden_deltas = [
            sum(
                output_deltas[j] * output_weights[j][i]
                for j in range(OUTPUT_SIZE)
            )
            * sigmoid_derivative(hidden_outputs[i])
            for i in range(HIDDEN_SIZE)
        ]

        for j in range(OUTPUT_SIZE):
            for i in range(HIDDEN_SIZE):
                output_weights[j][i] += (
                    learning_rate * output_deltas[j] * hidden_outputs[i]
                )
            output_bias[j] += learning_rate * output_deltas[j]

        for i in range(HIDDEN_SIZE):
            for k in range(INPUT_SIZE):
                hidden_weights[i][k] += (
                    learning_rate * hidden_deltas[i] * inputs[k]
                )
            hidden_bias[i] += learning_rate * hidden_deltas[i]


def train_model(epochs: int = EPOCHS, report_progress: bool = False) -> None:
    for epoch in range(epochs):
        train_epoch()
        if report_progress and epoch % 50 == 0:
            print(
                "Epoch:",
                epoch,
                "Accuracy:",
                round(get_accuracy(dataset), 2),
                "%",
            )


def predict(inputs: Sequence[float]) -> dict:
    if any(not math.isfinite(value) for value in inputs):
        raise ValueError("All input features must be finite numbers.")

    outputs, _ = forward(inputs)
    predicted_class = outputs.index(max(outputs))
    return {
        "prediction": CLASS_NAMES[predicted_class],
        "scores": dict(zip(CLASS_NAMES, outputs)),
    }


def run_worker() -> None:
    train_model()
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            result = predict(request["inputs"])
            response = {"id": request["id"], "result": result}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": str(error),
            }
        print(json.dumps(response, allow_nan=False), flush=True)


def run_console_training() -> None:
    train_model(report_progress=True)
    down_count = 0
    neutral_count = 0
    up_count = 0

    for inputs, _, _, _ in dataset:
        prediction = predict(inputs)["prediction"]
        if prediction == "Down":
            down_count += 1
        elif prediction == "Neutral":
            neutral_count += 1
        else:
            up_count += 1

    print("Down predictions:", down_count)
    print("Neutral predictions:", neutral_count)
    print("Up predictions:", up_count)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        run_worker()
    elif not sys.argv[1:]:
        run_console_training()
    else:
        raise SystemExit("Usage: python main.py [--worker]")
