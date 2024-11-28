# LLM Controllable Recipe Generation

With LLMs gaining more knowledge about different domains and industries, there are many applications in the nutrition space, more specifically recipe generation. People have traditionally relied on cookbooks and manual ways to generate recipes, but LLMs now have the capacity to generate equally as creative and coherent recipes in an automated fashion. Currently, tools like ChatGPT have already demonstrated the ability to generate a dish and provide a detailed recipe, given the names and quantities of ingredients. However, people like nutritionists and gym "rats" may want to generate recipes given calorie and macronutrient constraints to ensure they can design meal plans to achieve their health/fitness goals. In this project, I introduce a new task for recipe generation with nutritional constraints, fine-tune a pre-trained state-of-the-art LLM on this task, perform benchmarking and profiling of the training and inference pipelines, and conduct a thorough evaluation against baseline models (off-the-shelf SOTA LLMs).

This repository contains all the code used to train the recipe generation models and run evaluation. To get started, ensure to clone this repo and install the dependencies from `requirements.txt`. Below is a rundown of each file.

- `eda_preprocessing.ipynb`: contains all EDA experiments and preprocessing to generate train and test sets.
- `train_t5.py`: contains code to train T5 model (arguments on lines 223-235).
- `t5_hyperparameter_tuning.ipynb`: contains code for running hyperparameter tuning.
- `train.sh`: bash script to run t5 training with optimal set of hyperparameters.
- `generate_baseline_results.ipynb`: notebook to generate recipes on input using GPT 3.5 Turbo (baseline model).
- `evaluate.ipynb`: notebook to 1) generate t5 model outputs using different numbers of beams during decoding and 2) run evaluation on a variety of metrics.
- `demo.ipynb`: demo notebook using ipywidgets to allow users to 1) view the generated results on the train/test splits across the baseline and t5 models and 2) run inference on custom input.

*You can access the final T5 model [here](https://drive.google.com/drive/folders/1AHEgvAkE9JpBmOtIKqqHAb7MO5unwzwD?usp=sharing). This model has been fine-tuned using only caloric constraints in the input. Training a model using additional macronutrient constraints is a work in progress.*


