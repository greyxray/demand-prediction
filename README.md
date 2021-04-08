# Age segments
## Prepare clean python env

```bash
pyenv install 3.8.3
pyenv virtualenv 3.8.3 myenvmtr
pyenv local myenvmtr
```

## Install

```bash

## git clone this repo

pip install -r requirements.txt
pip install -e .

```

## Run the solution

```bash

### Train and validate the model
python model/main.py --data path/to/data.csv

### Plot the learning curves
python model/learning_curves.py --data path/to/data.csv
```
