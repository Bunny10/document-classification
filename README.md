# Document Classification

Document classification with PyTorch. This repository was made using the [practicalAI boilerplate](https://github.com/practicalAI/boilerplate) template.

### Set up with virtualenv
```
cd src
virtualenv -p python3 venv
source venv/bin/activate
python3 setup.py develop
gunicorn --log-level ERROR --workers 4 --timeout 60 --bind 0.0.0.0:5000 --access-logfile - --error-logfile - --reload wsgi
```
```
tensorboard --logdir="tensorboard" --port=6006
```

### Set up with docker
```bash
docker build -t document-classification:latest -f Dockerfile .
docker run -d -p 5000:5000 --name document-classification document-classification:latest
```

### Train a model
- Training `POST /train`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/train \
     --header "Content-Type: application/json" \
     --data '{
        "config_file": "training.json"
        }'
```

### Usage
- Inference `POST /predict`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/predict/latest \
     --header "Content-Type: application/json" \
     --data '{
        "X": "Global warming is inevitables, scientists warn."
        }'
```
- Python package
```python
from api.utils import predict
X = "Global warming is inevitables, scientists warn."
prediction = predict(experiment_id="latest", X=X)["data"]["prediction"]

>>> print (prediction)
[{'y': 'Sci/Tech', 'probability': 0.6540133357048035}, {'y': 'Business', 'probability': 0.339420884847641}, {'y': 'World', 'probability': 0.003702996065840125}, {'y': 'Sports', 'probability': 0.002862769179046154}]
```

### API endpoints
- Health check `GET /api`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification
```

- Training `POST /train`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/train \
     --header "Content-Type: application/json" \
     --data '{
        "config_file": "training.json"
        }'
```

- Inference `POST /predict`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/predict/latest \
     --header "Content-Type: application/json" \
     --data '{
        "X": "Global warming is inevitables, scientists warn."
        }'
```

- List of experiments `GET /experiments`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/experiments
```

- Experiment info `GET /info/<experiment_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/info
```

- Get classes for a model `GET /classes/<experiement_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/classes
```

- Delete an experiment `GET /delete/<experiement_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/delete/2019-03-14T01:05:49.989428_fafe6eb4-462f-11e9-bfe0-f0189887caab
```

### Directory structure
```
document-classification/
├── src/
|   ├── api/                      - holds all API scripts
|   |   ├── endpoints.py            - API endpoint definitions
|   |   └── utils.py                - utility functions for endpoints
|   ├── configs/                  - configuration files
|   |   ├── logging.json            - logger configuration
|   |   └── training.json           - training configuration
|   ├── datasets/                 - directory to hold datasets
|   |   └── news.csv                - data file
|   ├── document_classification/  - ML files
|   |   ├── dataset.py              - dataset
|   |   ├── model.py                - model functions
|   |   ├── utils.py                - utility functions
|   |   ├── vectorizer.py           - vectorize the processed data
|   |   └── vocabulary.py           - vocabulary to vectorize data
|   ├── tests/                    - tests
|   |   ├── e2e/                    - integration tests
|   |   ├── unit/                   - unit tests
|   ├── application.py            - application script
|   ├── config.py                 - application configuration
|   ├── requirements.txt          - python package requirements
|   ├── setup.py                  - custom package setup
|   ├── wsgi.py                   - application initialization
├── .dockerignore             - dockerignore file
├── .gitignore                - gitignore file
├── Dockerfile                - Dockerfile for the application
├── CODE_OF_CONDUCT.md        - code of conduct
├── CODEOWNERS                - code owner assignments
├── LICENSE                   - license description
└── README.md                 - repository readme
```
