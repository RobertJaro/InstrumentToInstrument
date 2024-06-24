import argparse
import os
import sys
from ftplib import FTP

import pandas as pd


def searchHalpha(base_dir=""):
    print("SEARCHING:", base_dir)
    files = ftp.nlst(base_dir)
    for file in files:
        if os.path.splitext(file)[1] != "":
            if file.endswith(".fts.gz") and (valid_files is not None and
                                             (valid_files.file_name == os.path.basename(file)).any()):
                downloadFile(file)
            continue
        if file == base_dir:
            continue
        searchHalpha(file)


def downloadFile(filename):
    target_file_path = os.path.join(local_path, os.path.basename(filename))
    if os.path.exists(target_file_path):
        return
    print("DOWNLOADING", filename)
    file = open(target_file_path, 'wb')
    ftp.retrbinary('RETR ' + filename, file.write)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download KSO Halpha data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--download_files_csv', type=str, help='csv file with basenames of files to download.',
                        required=False, default=None)
    args = parser.parse_args()

    local_path = args.download_dir
    valid_files = pd.read_csv(args.download_files_csv, index_col=0) if args.download_files_csv is not None else None

    os.makedirs(local_path, exist_ok=True)

    ftp = FTP('ftp.kso.ac.at')
    ftp.login('download', '9521treffen')
    searchHalpha("halpha4M/FITS/synoptic")
    ftp.quit()
