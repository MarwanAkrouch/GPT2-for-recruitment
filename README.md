# GPT2 for recruitment

GPT2 for recruitment is a project designed to evaluate the similarity between job descriptions (JDs) and resumes (CVs) using a GPT-based model. This repository contains the code for training qnd evaluating the similarity scores between JDs and CVs. There's a demo app for the POC.

## Project Structure

```
app.py
checkpoints/
    gpt2_b1024_l12_h12_e768_nowd_lt0.4_ht0.75/
    v0/gpt2_b1024_l12_h12_e768_slightbalancedataset/
data/
    test.csv
    train.csv
data.py
EDA/
    EDA_similarity.py
    EDA_similarity.txt
    EDA.ipynb
environment.yml
evaluate.ipynb
evaluate.py
evaluation_results/
main.py
metrics.py
minGPT/      -> minGPT model
model.py
predict.py
runs/        ->  tensorboard logs
trainer.py
utils.py     ->  checkpoint saving and loading
```

## Dependencies

1. Install dependencies from environment.yml

    ```sh
    conda env create -f environment.yml
    conda activate Edwin
    ```

## Run Demo App

2. run streamlit app

    ```sh
    streamlit run app.py
    ```

## Usage

### Data
Dataset from https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit

### Training

To train the model, run:
```sh
python main.py
```

### Saved models
model name meaning : gpt2_b1024_l12_h12_e768_nowd_lt0.4_ht0.75
  - gpt2_b1024_l12_h12_e768 : architecture details (n_layer, n_head, n_embd)
  - nowd : weight decay lambda set to 1e-6
  - lt : low threshold, score less than lt is no fit
  - ht : high threshol, score more than ht is good fit

### Tensorboard

To visualize training plots, run:
```sh
tensorboard --logdir=runs --port=6007
```
Note: there was data leakage in v0 models: val metrics aren't viable

### Evaluation

To evaluate the model, run:
```sh
python evaluate.py
```
check file for args

## Acknowledgements

This project uses the minGPT library by Andrej Karpathy. See the [minGPT repository](https://github.com/karpathy/minGPT) for more details.
