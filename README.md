Author: Philipp Würfel
Mail: PhilippWuerfel@live.de
DagsHub: https://dagshub.com/PhilippWuerfel/examen-dvc

# DVC & Dagshub

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```

## DVC Commands and typical workflow together with git
### Initialize dvc
- <code>dvc init</code>

### Add folders or files to be tracked by dvc
- make sure to remove from gitignore as dvc will configure gitignore

- <code>dvc add data/raw_data</code>

### Push changes to dvc
- e.g. do this after you've added new folder or file to dvc
- <code>dvc push</code>

### Commit changes to dvc when files change
- make sure to push after commit
- <code>dvc commit</code>

### Pull data
- e.g. when you want to get files from remote storage
- Note: pull = fetch + checkout
- <code>dvc pull</code>
- alternative:
  - <code>dvc fetch</code> (retrieve files into dvc cache)
  - <code>dvc checkout</code> (copy files from cache to folder)

### Track data changes together with git tags
- Changes happened e.g. updated processing steps of data
- Workflow
  1. dvc commit
  2. git > add, commit, tag, push
  3. dvc > push
  4. git > push
- <code>dvc commit</code>
- <code>git add --all</code>
- <code>git commit -m "Update and execution of preprocessing of data"</code>
- <code>git tag -a preprocess_001 -m "First version of data split, normalisation, model selection, hyperparameter tuning, model training and evaluation"</code>
- <code>git push origin --tags</code>
- <code>dvc push</code>
- <code>git push</code>

## DVC Pipeline
### Configure/Create a dvc pipeline
- create dvc.yaml file (you can also create yml file with <code>dvc stage</code> command)

```yaml
stages:
  ml_pipeline:
    cmd: python src/main.py
    deps:
    - data/raw_data
    - src/main.py
    - src/data/data_read.py
    - src/data/data_split.py
    - src/data/data_normalisation.py
    - src/models/lazy_model_selection.py
    - src/models/grid_search.py
    - src/models/model_training.py
    - src/models/model_evaluation.py
    outs:
    - data/joblib_data
    - data/processed_data
    - metrics
    - models
```

- cmd: bash commands to execute the pipeline
- deps: dependencies of the pipeline (scripts, folders, files)
- outs: outputs of pipeline (files, folders)

### Trigger dvc pipeline
<code>dvc repro</code>

### Create dvc pipeline graph
<code>dvc dag</code>