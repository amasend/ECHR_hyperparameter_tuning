import os
import subprocess
import sys

_DATASETS_PATH_ = './build/echr_database/datasets_documents/'
_PROCESSED_PATH_ = './build/echr_database/processed_documents'
_RAW_DOC_PATH_ = './build/echr_database/raw_normalized_documents'
_CONFIG_PATH_ = './config'
_OUTPUT_FOLDER_ = './cases'


def call_and_print(cmd):
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)
    while True:
        out = p.stderr.read(1)
        if out == '' and p.poll() is not None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()


def remove_dir(dir_):
    try:
        shutil.rmtree(dir_)
    except Exception as e:
        print(e)


def make_dir(dir_):
    try:
        os.makedirs(dir_)
    except Exception as e:
        print(e)


def make_case(n_range):
    make_dir(_OUTPUT_FOLDER_)

    for i in range(2, n_range+1):
        remove_dir(_RAW_DOC_PATH_)
        with open('{}/config.json'.format(_CONFIG_PATH_), 'w') as f:
            if i == 2:
                f.write("""{{
    "ngrams": {{
        "1": 1,
        "2": 1
    }}
}}""")
            elif i == 3:
                f.write("""{{
    "ngrams": {{
        "1": 1,
        "2": 1,
        "3": 1
    }}
}}""")
            elif i == 4:
                f.write("""{{
    "ngrams": {{
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1
    }}
}}""")
            elif i == 5:
                f.write("""{{
    "ngrams": {{
        "1": 1,
        "2": 1
        "3": 1,
        "4": 1,
        "5": 5
    }}
}}""")
            else:
                f.write("""{{
    "ngrams": {{
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "6": 1
    }}
}}""")

        call_and_print('python build.py')

        out_path = '{}/n_grams_{}'.format(_OUTPUT_FOLDER_, i)
        make_dir(out_path)
        #cmd = 'mv {}{} {}'.format(_DATASETS_PATH_, '*', out_path)
        cmd = 'mv {} {}/'.format(_PROCESSED_PATH_, out_path)
        call_and_print(cmd)
        cmd = 'mv {} {}/'.format(_RAW_DOC_PATH_, out_path)
        call_and_print(cmd)


def change_tokens(n_range, k_range):

    for n in range(1, n_range+1):
        for k in k_range:
            remove_dir(_RAW_DOC_PATH_)
            raw_doc = 'cp -r {}/n_grams_{}/raw_normalized_documents {}'.format(_OUTPUT_FOLDER_, n, _RAW_DOC_PATH_)
            call_and_print(raw_doc)
            call_and_print('python build.py --tokens {}'.format(k))
            out_path = '{}/n_grams_{}/k_{}'.format(_OUTPUT_FOLDER_, n, k)
            make_dir(out_path)
            cmd = 'mv {} {}/'.format(_DATASETS_PATH_, out_path)
            call_and_print(cmd)


if __name__ == "__main__":
    
    make_case(6)
    change_tokens(6, [1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000])
