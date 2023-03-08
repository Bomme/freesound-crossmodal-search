# Freesound Cross-Modal Search

Find sounds in Freesound with a text-to-audio retrieval system

## Get started

```bash
poetry shell
poetry install
python3 main.py
```

## Model training

```bash
python3 main.py fit --model config/model/default.yaml --data config/data/clotho.yaml --trainer ...
```
