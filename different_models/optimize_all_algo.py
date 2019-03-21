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
from multiprocessing import Process, Lock

# os.environ["OPENBLAS_MAIN_FREE"] = "1"


def optimize_in_model(models=None, lock=None):
    flavors = ['desc', 'bow', 'desc+bow']
    articles = ['1', '2', '3', '5', '6', '8', '10', '11', '13', '34', 'p1']

    for flavor in flavors:
        for article in articles:
            for model, path in models.items():
                parser = argparse.ArgumentParser(description='Optimizer script to select particular '
                                                             'worker and run hyperparameter tuning on these models.')
                parser.add_argument('--min_budget', type=float, help='Minimum number of epochs/k-fold.', default=20)
                parser.add_argument('--max_budget', type=float, help='Maximum number of epochs/k-fold.', default=50)
                parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
                                    default=10)
                parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true',
                                    default=False)
                parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. '
                                                               'An easy option is to use the job id of the clusters scheduler.',
                                    default=0)
                parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.',
                                    default=ni.interfaces()[0])
                parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, '
                                                                         'e.g. a NFS share.',
                                    default=path)
                parser.add_argument('--article', type=str, help='EHCR article to perform classification on.',
                                    default=article)
                parser.add_argument('--flavor', type=str, help='EHCR flavor to perform classification on.',
                                    default=flavor)
                parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=10)

                args=parser.parse_args()



                if 'dnn' in args.shared_directory:
                    from model_workers import KerasWorker as worker
                elif 'random_forest' in args.shared_directory:
                    from model_workers import RandomForestWorker as worker
                elif 'xgboost' in args.shared_directory:
                    from model_workers import XGBoostWorker as worker
                elif 'svm' in args.shared_directory:
                    from model_workers import SVMWorker as worker
                elif 'extratrees' in args.shared_directory:
                    from model_workers import ExtraTreesWorker as worker
                else:
                    raise Exception("""Please specify correct worker directory.
                    See possible choices:
                    - .\\results\\dnn\\
                    - .\\results\\random_forest\\
                    - .\\results\\xgboost\\
                    - .\\results\\svm\\
                    - .\\results\\extratrees\\""")
                from socket import gaierror

                for inter in ni.interfaces():
                    try:
                        # Every process has to lookup the hostname
                        host = hpns.nic_name_to_host(inter)
                        # host = '127.0.0.1'

                        if args.worker:
                            import time
                            time.sleep(5)   # short artificial delay to make sure the nameserver is already running
                            w = worker(run_id=args.run_id, host=host, timeout=120, article=args.article, flavor=args.flavor)
                            w.load_nameserver_credentials(working_directory=args.shared_directory)
                            w.run(background=False)
                            exit(0)

                        # This example shows how to log live results. This is most useful
                        # for really long runs, where intermediate results could already be
                        # interesting. The core.result submodule contains the functionality to
                        # read the two generated files (results.json and configs.json) and
                        # create a Result object.
                        result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)

                        # Start a nameserver:
                        NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
                        ns_host, ns_port = NS.start()

                        break
                    except gaierror:
                        continue
                # Start local worker
                workers = []
                for i in range(args.n_workers):
                    w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port,
                               timeout=120, article=args.article, flavor=args.flavor, id=i)
                    w.run(background=True)
                    workers.append(w)


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

                res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

                # store results
                with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
                    pickle.dump(res, fh)

                # shutdown
                bohb.shutdown(shutdown_workers=True)
                NS.shutdown()
                # Keep track of the best results (lock shared folders and files until process finishes)
                # lock.acquire()
                store_best_result_and_config(model_path=path, article=article, n_gram=4,  tokens=5000,
                                             flavor=flavor,  preprocessed=0,  pipeline=None)
                move_results_to_storage(n_gram=4, tokens=5000, article=article,
                                        flavor=flavor, model=path.split('\\')[-2])
                # lock.release()


if __name__ == "__main__":
    """Use parallelism based on particular ML model. 5 parallel processes with 
    hyperparameter tuning (5 different models)"""

    models = [{'extratrees': '.\\results\\extratrees\\'},
              {'random_forest': '.\\results\\random_forest\\'},
              {'dnn': '.\\results\\dnn\\'},
              {'xgboost': '.\\results\\xgboost\\'},
              {'svm': '.\\results\\svm\\'}]

    # procs = []
    # lock = Lock()
    # # instantiating process with arguments
    # for model in models:
    #     proc = Process(target=optimize_in_model, args=(model,lock))
    #     procs.append(proc)
    #     proc.start()
    #
    # # complete the processes
    # for proc in procs:
    #     proc.join()
    optimize_in_model(models[1], None)
