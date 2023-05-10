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

### WINE

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

### DIABETES

```shell
docker exec -it narlab-ml-1 mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///diabetes.db --default-artifact-root file:///mlruns
```

> http://localhost:8902

```shell
docker exec -it narlab-ml-1 mlflow ui --backend-store-uri sqlite:///diabetes.db --host 0.0.0.0 --port 5001
```

> http://localhost:8903

```shell
docker exec -it narlab-ml-1 /bin/bash
cd narlab/diabetes
python train.py 0.02 0.2
```

```shell
docker exec -it narlab-ml-1 mlflow run narlab/diabetes -P alpha=0.4 --env-manager local
```

```shell
docker exec -it narlab-ml-1 mlflow models serve --model-uri runs:/<run-id>/model
```
> `docker exec -it narlab-ml-1 mlflow models serve --host 0.0.0.0 --port 5002 --model-uri runs:/7d5d2a59e3a64b8e8ac352949ab90f59/model --env-manager local`

```script
curl -d '{"dataframe_split": {"columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"], "index": [0, 1, 2], "data": [[0.0380759064, 0.0506801187, 0.0616962065, 0.0218723855, -0.0442234984, -0.0348207628, -0.0434008457, -0.002592262, 0.0199074862, -0.0176461252], [-0.0018820165, -0.0446416365, -0.0514740612, -0.0263275281, -0.0084487241, -0.0191633397, 0.0744115641, -0.0394933829, -0.0683315471, -0.0922040496], [0.0852989063, 0.0506801187, 0.0444512133, -0.0056704223, -0.0455994513, -0.0341944659, -0.0323559322, -0.002592262, 0.0028613093, -0.025930339]]}}'  -H 'Content-Type: application/json' localhost:8904/invocations
```
