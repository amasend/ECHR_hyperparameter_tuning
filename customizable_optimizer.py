import os
import pickle
import argparse
import netifaces as ni
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import logging
logging.basicConfig(level=logging.DEBUG)
from store_best import store_best_result_and_config
from store_all_results import move_results_to_storage
import psutil
from socket import gaierror


def optimize_in_model(path, args, n_gram, tokens):
    articles = ['1', '2', '3', '5', '6', '8', '10', '11', '13', '34', 'p1']

    for article in articles:
        if 'dnn' in path:
            from model_workers import KerasWorker as worker
        elif 'random_forest' in path:
            from model_workers import RandomForestWorker as worker
        elif 'xgboost' in path:
            from model_workers import XGBoostWorker as worker
        elif 'svm' in path:
            from model_workers import SVMWorker as worker
        elif 'extratrees' in path:
            from model_workers import ExtraTreesWorker as worker

        for inter in ni.interfaces():
            try:
                # Every process has to lookup the hostname
                host = hpns.nic_name_to_host(inter)

                if args.worker:
                    import time
                    time.sleep(5)   # short artificial delay to make sure the nameserver is already running
                    w = worker(run_id=args.run_id, host=host, timeout=120, article=article, flavor=args.flavor)
                    w.load_nameserver_credentials(working_directory=path)
                    w.run(background=False)
                    exit(0)

                # This example shows how to log live results. This is most useful
                # for really long runs, where intermediate results could already be
                # interesting. The core.result submodule contains the functionality to
                # read the two generated files (results.json and configs.json) and
                # create a Result object.
                result_logger = hpres.json_result_logger(directory=path, overwrite=True)

                # Start a nameserver:
                NS = hpns.NameServer(run_id=args.run_id, host=host, port=0,
                                     working_directory=path)
                ns_host, ns_port = NS.start()
                break
            except gaierror:
                continue

        # Start local worker
        w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port,
                   timeout=120, article=article, flavor=args.flavor, id=1)
        w.run(background=True)


        # Run an optimizer
        bohb = BOHB(configspace=worker.get_configspace(),
                    run_id=args.run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    result_logger=result_logger,
                    min_budget=args.min_budget,
                    max_budget=args.max_budget,
                    )

        res = bohb.run(n_iterations=args.n_iterations)

        # store results
        with open(os.path.join(path, 'results.pkl'), 'wb') as fh:
            pickle.dump(res, fh)

        # shutdown
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        # Keep track of the best results
        store_best_result_and_config(model_path=path, article=article, n_gram=n_gram,  tokens=tokens,
                                     flavor=args.flavor,  preprocessed=0,  pipeline=None)
        move_results_to_storage(n_gram=n_gram, tokens=tokens, article=article,
                                flavor=args.flavor, model=path.split('/')[-2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimizer script to select particular '
                                                 'worker and run hyperparameter tuning on the model.')
    # The default parameters for min and max budget are case specific (for this case, shouldn't be changed)
    # For more information look at "model_workers.py" file to see dependencies
    parser.add_argument('--min_budget', type=float, help='Minimum number of epochs/k-fold.', default=400)
    parser.add_argument('--max_budget', type=float, help='Maximum number of epochs/k-fold.', default=5000)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
                        default=10)
    # TO DO: Hard to tell wat should be changed to work with this option... to further investigation
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true',
                        default=False)
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. '
                                                   'An easy option is to use the job id of the clusters scheduler.',
                        default=0)
    parser.add_argument('--model', type=str, help='ML algorithm name to use.',
                        choices=['extratrees', 'random_forest', 'dnn', 'xgboost', 'svm'])
    parser.add_argument('--flavor', type=str, help='EHCR flavor to perform classification on.',
                        choices=['desc', 'bow', 'desc+bow'])
    parser.add_argument('--n_gram', type=int, help='N-grams number',
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--tokens', type=int, help='Tokens number',
                        choices=[1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000])
    parser.add_argument('--cpu', type=int, help='Number of CPU to use.')

    args = parser.parse_args()

    p = psutil.Process()
    p.cpu_affinity([args.cpu])

    models = {'extratrees': './results/{}/extratrees/'.format(args.flavor),
              'random_forest': './results/{}/random_forest/'.format(args.flavor),
              'dnn': './results/{}/dnn/'.format(args.flavor),
              'xgboost': './results/{}/xgboost/'.format(args.flavor),
              'svm': './results/{}/svm/'.format(args.flavor)}

    optimize_in_model(path=models[args.model], args=args, n_gram=args.n_gram, tokens=args.tokens)
