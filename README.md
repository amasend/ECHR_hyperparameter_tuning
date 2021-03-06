
# European Court of Human Rights OpenData ML Classification Hyperparameter Tuning  
## Paper that describes full experiment is under: Project_paper.docx
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
# Docker image
Full working environment is available with docker image:
```bash
sudo docker run -ti amasend/echr:version2
```  
Docker environment only includes OneDrive version of experiment. Local storage will be included in the next release.  
# Functionality
<img src="https://github.com/amasend/ECHR_hyperparameter_tuning/blob/master/mermaid_graph.PNG"/>    

# How to run from a repository?
 - Firstly create a directory that will be containing all the generated datasets.
```bash
mkdir ~/dataset_storage
```  
 - Download all datasets from:  (All datasets could be found on https://1drv.ms/f/s!AkC_hbDmloKIjHh80Vdqc9OUSnwX)
 - Create additional directory for temporary dataset storage where specific datasets will be placed during computation.
```bash
mkdir ~/data
```  
 - Export environmental variable pointing a data directory for ECHR adta loader.
```bash
export ECHR_OD_PATH=~/data
```  
 - Run optimization process. You can specify which model you want to evaluate, on which flavors, n_grams and tokens.
```bash
python run_optimization.py --flavors --models --n_grams --tokens --onedrive --dataset_storage
```  
 - flavors - ['desc', 'bow', 'desc+bow'] 
 - models - ['extratrees', 'random_forest', 'dnn', 'xgboost', 'svm']
 - n_grams - [1, 2, 3, 4, 5, 6] 
 - tokens - [1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000]
 - onedrive - True/False (if True use OneDrive storage, if False use local storage) 
 - dataset_storage - ~/dataset_storage (path to dataset storage)
  
Please have in mind that computation time for all combinations is really huge. By runnig this experiment with a default parameters you should be aware that each n_gram and token is computed separetly (one after another), only all models and flavors are multiprocessed.

# Generating results plots  
All components are under '/different_models' directory.  
After generating all results or partial results we can make plots of that results for better understanding.  
```bash
python show_results.py --results_path "path_to_results_folder" --model "model_name" --n_grams "number_of_grams" --tokens "number_of_tokens" --flavor "flavor_name" --metric "metric_name_to_display" --option "which_plot_to_display"
```  
There are couple of methods implemented for user use and based on that, we can plot as follows:  

 - best new results comparison per model and n-gram (after closing one plot, another will appear for the next n-gram)

```bash
python show_results.py --results_path "all_results" --model "random_forest" --metric "test_mean_c_MCC" --option "new_best_results"
```  
<img src="https://github.com/amasend/ECHR_hyperparameter_tuning/blob/master/plots/new_results_best.PNG"/>

 - old best results vs new results (old best among all models, new best only from one model), per n-gram

```bash
python show_results.py --results_path "all_results" --model "random_forest" --metric "test_mean_c_MCC" --option "old_vs_new" --n_grams 5
```  
<img src="https://github.com/amasend/ECHR_hyperparameter_tuning/blob/master/plots/old_best_vs_new.png"/>

 - normalized confusion matrices (only for new results)

```bash
python show_results.py --results_path "all_results" --model "random_forest" --metric "test_mean_c_MCC" --option "confusion_matrix" --n_grams 5 --tokens 5000
```  
<img src="https://github.com/amasend/ECHR_hyperparameter_tuning/blob/master/plots/confusion_matrix.png"/>

 - computing times (run times) per flavor and per optimizer run

```bash
python show_results.py --results_path "all_results" --model "random_forest" --metric "test_mean_c_MCC" --option "computing_times" --n_grams 5 --tokens 5000
```  
<img src="https://github.com/amasend/ECHR_hyperparameter_tuning/blob/master/plots/computing_times.png"/>
