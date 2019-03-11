# Document Classification

Document classification w/ PyTorch. This repository was made using the [practicalAI boilerplate](https://github.com/practicalAI/boilerplate).

### Set up with virtualenv
```
cd src
virtualenv -p python3 venv
source venv/bin/activate
python setup.py develop
gunicorn --log-level ERROR --workers 4 --bind 0.0.0.0:5000 --access-logfile - --error-logfile - --reload wsgi
```

### Set up with docker
```bash
docker build -t document-classification:latest -f Dockerfile .
docker run -d -p 5000:5000 --name document-classification document-classification:latest
docker exec -it document-classification /bin/bash
```

### Usage
- Inference `POST /predict`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/predict/latest \
     --header "Content-Type: application/json" \
     --data '{
        "X": "Global warming is an increasing threat and scientists are working to find a solution."
        }'
```
- Python package
```python
from api.utils import predict
experiment_id = "latest"
X = "Global warming is an increasing threat and scientists are working to find a solution."
predictions = predict(experiment_id, X)["data"]["predictions"]
print (predictions)
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
        "X": "Global warming is an increasing threat and scientists are working to find a solution."
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

- Performance across classes `GET /document-classification/performance/<experiment_id>`
```bash
curl --request GET \
     http://localhost:5000/document-classification/performance
```

- Delete an experiment `GET /delete/<experiement_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/delete/1552345515_21f4c3ae-4452-11e9-ab10-f0189887caab
```

### Directory structure
```bash
├── src/
|   ├── **api**: holds all API scripts
|   |   ├── *endpoints.py*: API endpoint definitions
|   |   └── *utils.py*: utility functions for endpoints
|   ├── **datasets**: directory to hold datasets
|   ├── **configs**: configuration files
|   |   ├── *logging.json*: logger configuration
|   |   ├── *training.json*: training configuration
|   ├── **document_classification**:
|   |   ├── *dataset.py*: dataset/dataloader
|   |   ├── *inference.py*: inference operations
|   |   ├── *load.py*: load the data
|   |   ├── *model.py*: model architecture
|   |   ├── *preprocess.py*: preprocess the data
|   |   ├── *split.py*: split the data
|   |   ├── *training.py*: train the model
|   |   ├── *utils.py*: utility functions
|   |   ├── *vectorizer.py*: vectorize the processed data
|   |   └── *vocabulary.py*: vocabulary to vectorize data
|   ├── *application.py*: application script
|   ├── *config.py*: application configuration
|   ├── *requirements.txt*: python package requirements
|   ├── *setup.py*: custom package setup
|   ├── *wsgi.py*: application initialization
|   ├── *.dockerignore*: dockerignore file
|   ├── *.gitignore*: gitignore file
|   ├── *Dockerfile*: Dockerfile for the application
|   ├── *CODE_OF_CONDUCT.md*: code of conduct
|   ├── *CODEOWNERS*: code owner assignments
|   ├── *LICENSE*: license description
|   └── *README.md*: repository readme
```