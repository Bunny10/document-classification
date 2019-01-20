# Document Classification

Document classification using PyTorch. This repository was made using the [productionML cookiecutter](https://github.com/practicalAI/productionML) template.

### Set up with virtualenv
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==1.0.0
python setup.py develop
cd document_classification
gunicorn --log-level ERROR --workers 4 --bind 0.0.0.0:5000 --access-logfile - --error-logfile - --reload wsgi
```

### Set up with docker
```bash
docker build -t document_classification:latest --build-arg DIR="$PWD" -f Dockerfile .
docker run -d -p 5000:5000 --name document_classification document_classification:latest
docker exec -it document_classification /bin/bash
``

### API endpoints
- Health check `GET /api`
```bash
curl -X GET \
     http://localhost:5000/ \
     -H "Content-Type: application/json"
```

- Training `POST /train`
```bash
curl -X POST \
     http://localhost:5000/train \
     -H "Content-Type: application/json" \
     -d '{
        "config_filepath": "/Users/goku/Documents/document_classification/configs/train.json"
        }'
```

- Inference `POST /infer`
```bash
curl -X POST \
     http://localhost:5000/infer \
     -H "Content-Type: application/json" \
     -d '{
        "experiment_id": "latest",
        "X": "Global warming is an increasing threat and scientists are working to find a solution."
        }'
```

- List of experiments `GET /experiments`
```bash
curl -X GET \
     http://localhost:5000/experiments \
     -H "Content-Type: application/json"
```

- Experiment info `GET /info/<experiment_id>`
```bash
curl -X GET \
     http://localhost:5000/info/latest \
     -H "Content-Type: application/json"
```

- Delete an experiment `GET /delete/<experiement_id>`
```bash
curl -X GET \
     http://localhost:5000/delete/1545593561_8371ca74-06e9-11e9-b8ca-8e0065915101 \
     -H "Content-Type: application/json"
```

- Get classes for a model `GET /classes/<experiement_id>`
```bash
curl -X GET \
     http://localhost:5000/classes/latest \
     -H "Content-Type: application/json"
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
