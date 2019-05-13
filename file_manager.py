import os
import zipfile
import shutil
import stat


class FileManager:
    """Class to interact with OneDrive API."""
    def __init__(self, dataset_storage_path):
        self.data_path = os.environ['ECHR_OD_PATH']
        self.dataset_storage = dataset_storage_path
        self.articles = ['article_1.zip', 'article_2.zip', 'article_3.zip', 'article_5.zip', 'article_6.zip',
                         'article_8.zip', 'article_10.zip', 'article_11.zip', 'article_13.zip', 'article_34.zip',
                         'article_p1.zip']
        self.n_gram = None
        self.token = None

    def move_data(self, n_gram, token):
        """Method to move datasets into a destination folder for further processing."""
        print('Copying dataset into ECHR storage.')
        self.n_gram = '{}_n_grams'.format(n_gram)
        self.token = 'k_{}'.format(token)
        for article in self.articles:
            # Copy datasets in .zip format into our destination
            shutil.copyfile(os.path.join(self.dataset_storage,
                                         self.n_gram,
                                         self.token,
                                         'datasets_documents',
                                         article), '{}{}'.format(self.data_path, article))

            # Unzip
            zip_ref = zipfile.ZipFile('{}{}'.format(self.data_path, article), 'r')
            zip_ref.extractall('{}{}'.format(self.data_path, article.split('.')[0]))
            zip_ref.close()
            # Delete zipped file
            self.remove_file('{}{}'.format(self.data_path, article))
            print('{} {} {} copied'.format(self.n_gram, self.token, article))

    def remove_data(self):
        """Clean data folder, prepare for new data."""
        os.chmod(self.data_path[:-1], stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        self.remove_dir(self.data_path)
        self.make_dir(self.data_path)

    @staticmethod
    def remove_file(file_):
        try:
            os.remove(file_)
            print("File Removed! {}".format(file_))
        except Exception as e:
            print(e)
            exit()

    @staticmethod
    def remove_dir(dir_):
        try:
            shutil.rmtree(dir_)
        except Exception as e:
            print(e)
            exit()

    @staticmethod
    def make_dir(dir_):
        try:
            os.makedirs(dir_)
        except Exception as e:
            print(e)
            exit()