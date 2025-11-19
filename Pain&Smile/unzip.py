import os
from zipfile import ZipFile

with ZipFile('dataset.zip') as myzip:
    myzip.extractall()
    print('Extracting Complete.')


