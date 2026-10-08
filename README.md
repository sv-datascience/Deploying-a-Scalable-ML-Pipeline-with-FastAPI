# Deploying a Machine Learning Model with FastAPI

**GitHub Repository:** https://github.com/sv-datascience/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

WGU D501 Machine Learning DevOps project. This project trains a Gradient Boosting classifier on U.S. Census data to predict whether income is >50K or <=50K, evaluates the model on slices of the data, and serves predictions through a RESTful API built with FastAPI. GitHub Actions runs flake8 and pytest on every push.

## Project Structure
* `data/census.csv` - Census Income dataset
* `ml/data.py` - data preprocessing (`process_data`, `apply_label`)
* `ml/model.py` - train, inference, metrics, save/load, and slice performance functions
* `train_model.py` - ML pipeline: loads data, splits, trains, saves the model and encoder, and writes `slice_output.txt`
* `model/` - saved `model.pkl` and `encoder.pkl`
* `test_ml.py` - unit tests
* `main.py` - FastAPI app (GET `/` and POST `/data/`)
* `local_api.py` - sends a GET and a POST request to the running API
* `model_card.md` - model card
* `slice_output.txt` - model performance on categorical slices
* `screenshots/` - `continuous_integration.png`, `unit_test.png`, `local_api.png`

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip

## How to Run
```bash
python train_model.py          # train, save model/encoder, write slice_output.txt
pytest test_ml.py -v           # run unit tests
uvicorn main:app --reload      # start the API (terminal 1)
python local_api.py            # send GET and POST requests (terminal 2)
```
