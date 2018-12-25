# Document Classification

### Set up with virtualenv
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
python document_classification/application.py
```

### API endpoints
- Health check `GET /api`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/
```

- Training `POST /train`
```bash
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"config_filepath": "/Users/goku/Documents/document_classification/configs/train.json"}' \
     http://localhost:5000/train
```

- Inference `POST /infer`
```bash
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"config_filepath": "/Users/goku/Documents/document_classification/configs/infer.json"}' \
     http://localhost:5000/infer
```

- List of experiments `GET /experiments`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/experiments
```

- Experiment info `GET /info/<experiment_id>`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/info/latest
```

- Delete an experiment `GET /delete/<experiement_id>`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/delete/1545593561_8371ca74-06e9-11e9-b8ca-8e0065915101
```

- Get classes for a model `GET /classes/<experiement_id>`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/classes/latest
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
- Dockerfile
- Swagger API documentation
- serving with Onnx and Caffe2

