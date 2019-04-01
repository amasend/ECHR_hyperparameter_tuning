import os
import pprint
import zipfile
import onedrivesdk
import shutil
import time


class OneDriveAPI:
    """Class to interact with OneDrive API."""
    def __init__(self, get_ids=True):
        self.client_id='6a2475f8-6ae9-4c3d-a115-2a67794947df'
        self.api_base_url='https://api.onedrive.com/v1.0/'
        self.scopes=['wl.signin', 'wl.offline_access', 'onedrive.readwrite']
        self.base_url='https://api.onedrive.com/v1.0/'
        self.data_path = os.environ['ECHR_OD_PATH']
        self.one_drive_structure = {}
        self.dataset = 'datasets_documents'
        self.articles = ['article_1.zip', 'article_2.zip', 'article_3.zip', 'article_5.zip', 'article_6.zip',
                         'article_8.zip', 'article_10.zip', 'article_11.zip', 'article_13.zip', 'article_34.zip',
                         'article_p1.zip']
        self.n_gram = None
        self.token = None

        # OAuth part to One Drive
        while True:
            try:
                self.http_provider = onedrivesdk.HttpProvider()
                self.auth_provider = onedrivesdk.AuthProvider(self.http_provider, self.client_id, self.scopes)
                self.auth_provider.load_session()
                self.auth_provider.refresh_token()
                self.client = onedrivesdk.OneDriveClient(self.base_url, self.auth_provider, self.http_provider)
                # Check files IDs
                if get_ids:
                    self.get_files_ids()
                break
            except Exception as e:
                print(e)
                continue

    def get_files_ids(self):
        """Check files IDs on One Drive and create local dictionary to further compute."""
        collection = self.client.item(drive='me', id='888296E6B085BF40!1656').children.request(top=100).get()
        for n_gram, id_1 in [(x.name, x.id) for x in collection]:
            self.one_drive_structure[n_gram] = {'id': id_1,
                                                'tokens': {}}
            for token, id_2 in [(x.name, x.id) for x in self.client.item(drive='me',
                                                                         id=id_1).children.request(top=100).get()]:
                self.one_drive_structure[n_gram]['tokens'][token] = {'id': id_2,
                                                                     'dataset': {}}
                dataset, id_3 = [(x.name, x.id) for x in self.client.item(drive='me',
                                                                          id=id_2).children.request(top=100).get()][0]
                self.one_drive_structure[n_gram]['tokens'][token]['dataset'][dataset] = {'id': id_3,
                                                                                         'article': {}}
                for article, id_4 in [(x.name, x.id) for x in self.client.item(drive='me',
                                                                               id=id_3).children.request(top=100).get()]:
                    self.one_drive_structure[n_gram]['tokens'][token]['dataset'][dataset]['article'][article] = {'id': id_4}
        pprint.pprint(self.one_drive_structure)

    def download_data(self, n_gram, token):
        """Method to download article from OneDrive (in .zip), unzip, delete zipped file."""
        self.n_gram = '{}_n_grams'.format(n_gram)
        self.token = 'k_{}'.format(token)
        for article in self.articles:
            try:
                id = self.one_drive_structure[self.n_gram]['tokens'][self.token]['dataset'][self.dataset]['article'][article]['id']
            except KeyError as e:
                raise KeyError('There is no data for {} {}'.format(self.n_gram, self.token))
            print('{} {} {} downloading in progress...'.format(self.n_gram, self.token, article))
            # Download
            self.client.item(drive='me', id=id).download('{}{}'.format(self.data_path, article))
            # Unzip
            zip_ref = zipfile.ZipFile('{}{}'.format(self.data_path, article), 'r')
            zip_ref.extractall('{}{}'.format(self.data_path, article.split('.')[0]))
            zip_ref.close()
            # Delete zipped file
            self.remove_file('{}{}'.format(self.data_path, article))
            print('{} {} {} downloaded'.format(self.n_gram, self.token, article))

    def upload_all_results(self):
        """Uploads all results to OneDrive"""
        try:
            shutil.make_archive('all_results_packed', 'zip', 'all_results')
            self.client.item(drive='me', id='888296E6B085BF40!1702').children['all_results.zip'].upload('.\\all_results_packed.zip')
            self.remove_file('.\\all_results_packed.zip')
        except Exception as e:
            print(e)

    def remove_data(self):
        """Clean data folder, prepare for new data."""
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


if __name__ == "__main__":
    """Use only for uploading results to One Driive (periodically)"""
    while True:
        file_manager = OneDriveAPI(get_ids=False)  # Initiate file manager with files IDs
        file_manager.upload_all_results()
        time.sleep(60*5)

