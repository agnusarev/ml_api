# ml_api
This FastApi API clusterring data from dataset German Credit Risk (see: https://www.kaggle.com/datasets/uciml/german-credit).
API uses KMeans method.

## Installation

````bash
poetry install
````

## Runs

````bash
make run_server
````

## Tests

````bash
make test
````

## Example request
````json
[
    {
        "id": 0,
        "age": 67,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "save_account": "unknown",
        "check_account": "little",
        "credit_amount": 1169,
        "duration": 6,
        "purpose": "radio\/TV"
    },
    {
        "id": 1,
        "age": 22,
        "sex": "female",
        "job": 2,
        "housing": "own",
        "save_account": "little",
        "check_account": "moderate",
        "credit_amount": 5951,
        "duration": 48,
        "purpose": "radio\/TV"
    }
]
````
