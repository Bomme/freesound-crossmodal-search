# Freesound Cross-Modal Search

Find sounds in Freesound with a text-to-audio retrieval system

## Get started

```bash
$ poetry shell
$ poetry install
$ python3 main.py --help
```

## Model training

```bash
$ python3 main.py fit --model config/model/default.yaml --data config/data/clotho.yaml --trainer ...
```

## Demo

```bash
$ streamlit run retrieval_interface/streamlit_app.py -- --help
$ streamlit run retrieval_interface/streamlit_app.py -- data_dir audio_dir ckpt_path
```
