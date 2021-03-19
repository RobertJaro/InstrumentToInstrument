import os
import sys
from ftplib import FTP


def searchHalpha(base_dir=""):
    print("SEARCHING:", base_dir)
    files = ftp.nlst(base_dir)
    for file in files:
        if os.path.splitext(file)[1] != "":
            if file.endswith(".fts.gz"):
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
    local_path = sys.argv[1]

    os.makedirs(local_path, exist_ok=True)

    ftp = FTP('ftp.kso.ac.at')
    ftp.login('download', '9521treffen')
    searchHalpha("halpha4M/FITS/synoptic")
    ftp.quit()
