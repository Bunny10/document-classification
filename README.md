# Document Classification

Document classification using PyTorch. This repository was made using the [productionML cookiecutter](https://github.com/practicalAI/productionML) template.

### Set up with virtualenv
```
cd src
virtualenv -p python3.6 venv
source venv/bin/activate
python setup.py develop
cd document_classification
gunicorn --log-level ERROR --workers 4 --bind 0.0.0.0:5000 --access-logfile - --error-logfile - --reload wsgi
```

### Set up with docker
```bash
docker build \
    --tag document_classification:latest \
    --file Dockerfile .
docker run \
    --detach \
    --publish 5000:5000 \
    --name document_classification \
    document_classification:latest
docker exec -it document_classification /bin/bash
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
        "config_file": "train.json"
        }'
```

- Inference `POST /infer`
```bash
curl --request POST \
     --url http://localhost:5000/document-classification/infer/latest \
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
     --url http://localhost:5000/document-classification/info/latest
```

- Get classes for a model `GET /classes/<experiement_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/classes/latest
```

- Performance across classes `GET /document-classification/performance/<experiment_id>`
```bash
curl --request GET \
     http://localhost:5000/document-classification/performance/latest
```

- Delete an experiment `GET /delete/<experiement_id>`
```bash
curl --request GET \
     --url http://localhost:5000/document-classification/delete/1551157471_006209fa-3984-11e9-95c0-8c8590964109
```

### Content
- **datasets**: directory to hold datasets
- **configs**: configuration files
    - *train.json*: training configurations
    - *infer.json*: inference configurations
- **document_classification**:
    - *application.py*: application script
    - *config.py*: application configuration
    - *utils.py*: application utilities
    - **api**: holds all API scripts
        - *api.py*: API call definitions
        - *utils.py*: utility functions
    - **ml**:
        - *dataset.py*: dataset/dataloader
        - *inference.py*: inference operations
        - *load.py*: load the data
        - *model.py*: model architecture
        - *preprocess.py*: preprocess the data
        - *split.py*: split the data
        - *training.py*: train the model
        - *utils.py*: utility functions
        - *vectorizer.py*: vectorize the processed data
        - *vocabulary.py*: vocabulary to vectorize data
- *.gitignore*: gitignore file
- *LICENSE*: license of choice (default is MIT)
- *requirements.txt*: python package requirements
- *setup.py*: custom package setup


### TODO
- experiment id validation
- HTTPStatus code (ex. HTTPSStatus.BAD_REQUEST)
- Dockerfile
- Swagger API documentation
- serving with Onnx and Caffe2
