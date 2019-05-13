import psutil
import subprocess as sub
from onedrive_api import OneDriveAPI
from file_manager import FileManager
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script for rnunning optimization.')
    parser.add_argument('--flavors', type=list, help='List of flavors to compute.',
                        default=['desc', 'bow', 'desc+bow'])
    parser.add_argument('--models', type=list, help='List of models to evaluate.',
                        default=['extratrees', 'random_forest', 'dnn', 'xgboost', 'svm'])
    parser.add_argument('--n_grams', type=list, help='List of N-grams to compute.',
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--tokens', type=list, help='List of tokens to compute.',
                        default=[1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000])
    parser.add_argument('--onedrive', type=bool, help='Use OneDrive storage?',
                        default=False)
    parser.add_argument('--dataset_storage', type=str, help='Local path to dataset storage.',
                        default='D:')

    args = parser.parse_args()

    _FLAVORS_ = args.flavors
    _MODELS_ = args.models
    _N_GRAMS_ = args.n_grams
    _TOKENS_ = args.tokens

    if args.onedrive:
        """The main process for file and optimizer management."""
        # Start periodic uploads to One Drive (all results)
        upload = sub.Popen('python onedrive_api.py', shell=True)
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

    else:
        file_manager = FileManager(dataset_storage_path=args.dataset_storage)
        for n_gram in _N_GRAMS_:
            for tokens in _TOKENS_:

                file_manager.remove_data()  # Clean data folder

                try:
                    file_manager.move_data(n_gram, tokens)  # Move new data
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
