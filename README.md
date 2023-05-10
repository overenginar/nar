# nar

```shell
mkdir ~/narlab_data
```

```shell
docker compose build
```

```shell
docker compose up --no-build
```

```shell
docker ps
```

```shell
docker exec -it narlab-ml-1 mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///wine.db --default-artifact-root file:///mlruns
```

> http://localhost:8902

```shell
docker exec -it narlab-ml-1 mlflow ui --backend-store-uri sqlite:///wine.db --host 0.0.0.0 --port 5001
```

> http://localhost:8903


```shell
docker exec -it narlab-ml-1 /bin/bash
cd narlab/wine
python train.py 0.02 0.2
```

```shell
docker exec -it narlab-ml-1 mlflow run narlab/wine -P alpha=0.4 --env-manager local
```

```shell
docker exec -it narlab-ml-1 mlflow models serve --model-uri runs:/<run-id>/model
```
> `docker exec -it narlab-ml-1 mlflow models serve --host 0.0.0.0 --port 5002 --model-uri runs:/ee03b82d9ed84ea1a4ede4cc8afe2206/model --env-manager local`

```script
curl -d '{"dataframe_split": {"columns":["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],"index":[0,1,2],"data":[[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8],[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5],[8.1,0.28,0.4,6.9,0.05,30.0,97.0,0.9951,3.26,0.44,10.1]]}}'  -H 'Content-Type: application/json' localhost:8904/invocations
```

```python

import json
import requests
import pandas as pd
url = f'http://localhost:5002/invocations'
headers = {'Content-Type': 'application/json',}

test_data = (
    pd.read_csv('narlab/wine/wine-quality.csv')
    .drop('quality', axis=1)
    .iloc[[0,1,2]]
    .to_json(orient='split')
)

http_data = json.dumps({"dataframe_split": json.loads(test_data)})

r = requests.post(url=url, headers=headers, data=http_data)
print(f'Predictions: {r.text}')

```