
# European Court of Human Rights OpenData ML Classification Hyperparameter Tuning  
Under construction...  

Docker image with early version available at: https://hub.docker.com/r/amasend/echr

It is recommendet to have an [Anaconda with python 3.5](https://www.anaconda.com/distribution/#download-section) environment installed with preinstalled data science packages like: pandas, numpy, scikit-learn etc.  
Dependencies:  
```bash
conda install tensorflow keras
```  
For hyperparameter tuning [HpBandSter](https://github.com/automl/HpBandSter) was used.  
```bash
pip install hpbandster
```  
To load the datasets [ECHR-OD_loader](https://github.com/aquemy/ECHR-OD_loader) was used with data storage environment variable named: *ECHR_OD_PATH*  

## Directory structure:
```bash
├── different_models
│   ├── results
│   │   ├── bow
│   │   │	├── dnn
│   │   │	├── extratrees
│   │   │	├── random_forest
│   │   │	├── svm
│   │   │	├── xgboost
│   │   ├── desc
│   │   │	├── ...
│   │   ├── desc_bow
│   │   │	├── ...
│   ├── all_results
│   │   ├── 1_gram/1000_tokens/1_article...
│   │   ├── 2_gram/1000_tokens/1_article...
│   │   ├── ...
│   ├── best_results.csv
│   ├── customizable_optimizer.py
│   ├── model_workers.py
│   ├── onedrive_api.py
│   ├── run_optimization.py
│   ├── store_all_results.py
│   ├── store_best.py
├── DNN
├── ExtraTrees
├── ...
├── ...
```
