import psutil
import subprocess as sub
from onedrive_api import OneDriveAPI

_FLAVORS_ = ['desc', 'bow', 'desc+bow']
_MODELS_ = ['extratrees', 'random_forest', 'dnn', 'xgboost', 'svm']
_N_GRAMS_ = [1, 2, 3, 4, 5, 6]
_TOKENS_ = [1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000]

if __name__ == "__main__":
    """The main process for file and optimizer management."""
    # Start periodic uploads to One Drive (all results)
    upload = sub.Popen('python onedrive_api.py', shell=True)
    upload.wait()
    for n_gram in _N_GRAMS_:
        for tokens in _TOKENS_:
            file_manager = OneDriveAPI()  # Initiate file manager with files IDs
            file_manager.remove_data()  # Clean data folder
            try:
                file_manager.download_data(n_gram, tokens)  # Download new data
                cpu_number = psutil.cpu_count()  # Distribute processes over entire CPU
                child_processes = []
                for flavor in _FLAVORS_:
                    for model in _MODELS_:
                        if cpu_number == 0:
                            cpu_number = psutil.cpu_count()
                        cpu_number -= 1
                        child_processes.append(sub.Popen('python customizable_optimizer.py --model {} '
                                                         '--flavor {} --n_gram {} '
                                                         '--tokens {} --cpu {}'.format(model, flavor,
                                                                                       n_gram, tokens, cpu_number),
                                                         shell=True))
                for proc in child_processes:
                    proc.wait()
            except KeyError as e:
                print(e)

