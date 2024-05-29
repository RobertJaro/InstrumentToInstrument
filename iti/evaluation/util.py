from dateutil.parser import parse
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from google.cloud.storage import Client, transfer_manager
from google.cloud import storage

from iti.translate import *
from iti.data.editor import NormalizeRadiusEditor, AIAPrepEditor, NormalizeExposureEditor, MapToDataEditor, \
    SWAPPrepEditor, LoadMapEditor, solo_norm, proba2_norm

from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

################################### Get Data ###################################

def getSWAPdata(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(resolution=1024).call(s_map)
    return s_map

def getAIAdata(f, resolution=2048):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(resolution=resolution).call(s_map)
    s_map = AIAPrepEditor(calibration='auto').call(s_map)
    #data, _ = MapToDataEditor().call(s_map)
    return s_map

def getFSIdata(f):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(resolution=1024, fix_irradiance_with_distance=True).call(s_map)
    return s_map

def getHRIdata(f, resolution=4096):
    s_map, _ = LoadMapEditor().call(f)
    s_map = NormalizeRadiusEditor(resolution=resolution, rotate_north_up=False).call(s_map)
    return s_map


################################### Translatiom  ###############################

def translate(files, translator):
    if torch.cuda.is_available():
        maps = list(translator.translate([f for f in files]))
    else:
        with Pool(4) as p:
            maps = list(tqdm(p.imap(translator, [f for f in files]), total=len(files)))
    return maps


################################### Intensity ##################################

def getIntensity(files, channels, editor):
    intensity = {}
    for c, c_files in zip(channels, files):
        #c_files = c_files[::len(c_files) // 10]
        dates = [parse(os.path.basename(f).replace('.fits', '')) for f in c_files]
        with Pool(4) as p:
            means = [np.nanmean(m) for m in tqdm(p.imap(editor, c_files), total=len(c_files))]
        intensity[c] = (dates, means)
    return intensity


def getITIIntensity(maps, channels):
    intensity = {}
    for c, map in zip(channels, maps):
        dates = [m.meta['date-obs'] for m in map]
        means = []
        for m in map:
            means.append(np.nanmean(m.data))
        intensity[c] = (dates, means[::len(c)])
    return intensity

################################### Evaluation ##################################

def difference_map(original, ground_truth, iti):
    baseline = np.abs(original.data - ground_truth.data) * 100
    iti = np.abs(iti.data - ground_truth.data) * 100
    return baseline, iti

################################### Plotting ####################################

def plotLightcurve(df_base, df_ground_truth, df_iti, name):
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    axs.plot(df_base['time'], df_base['intensity'], label='Baseline', c='g')
    axs.plot(df_iti['time'], df_iti['intensity'], label='ITI', c='r')
    axs.plot(df_ground_truth['time'], df_ground_truth['intensity'], label='Ground truth', c='b')
    axs.plot(df_ground_truth['time'], df_ground_truth['intensity'] + df_ground_truth['MA'].std(), 'b--')
    axs.plot(df_ground_truth['date'], df_ground_truth['intensity'] - df_ground_truth['intensity'].std(), 'b--')
    axs.fill_between(df_ground_truth['date'], df_ground_truth['intensity'] + df_ground_truth['intensity'].std(),
                     df_ground_truth['intensity'] - df_ground_truth['intensity'].std(), alpha=0.05, facecolor='blue')
    axs.ylabel('Intensity [DN/s]', fontsize=20)
    axs.xlabel('Time', fontsize=20)
    axs.title('Light curve', fontsize=40)
    axs.xticks(rotation=45, fontsize=15)
    axs.yticks(fontsize=15)
    axs.legend(fontsize="20")
    plt.savefig('Lightcurve_'+f'{name}'+'.jpg')


def plotImageComparison(original, ground_truth, iti, original_norm, ground_truth_norm, path=None, name=None):
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ground_truth}, figsize=(50, 20), dpi=100)
    original.plot(axes=axs[0], norm=original_norm)
    ground_truth.plot(axes=axs[1], norm=ground_truth_norm)
    iti.plot(axes=axs[2], norm=ground_truth_norm)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[0].set_title('Original', fontsize=70)
    axs[1].set_title('Ground Truth', fontsize=70)
    axs[2].set_title('ITI', fontsize=70)
    plt.savefig(os.path.join(path)+f'ImageComparison_{name}.jpg') if path & name is not None else plt.show()
    plt.close()


################################### Save to FITS ##################################

def saveToFITS(maps, path):
    for i, m in tqdm(enumerate(maps)):
        m.save(path+maps[i].meta['date-obs']+'.fits')


################################### Download GCP bucket ###################################

def download_gcp_bucket(bucket_name, bucket_directory = "", destination_directory="", workers=8, max_results=1000):

    """Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to which to download all of the files. This
    # string is prepended (with os.path.join()) to the name of each blob to form
    # the full path. Relative paths and absolute paths are both accepted. An
    # empty string means "the current working directory". Note that this
    # parameter allows accepts directory traversal ("../" etc.) and is not
    # intended for unsanitized end user input.
    # destination_directory = ""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    # The maximum number of results to fetch from bucket.list_blobs(). This
    # sample code fetches all of the blobs up to max_results and queues them all
    # for download at once. Though they will still be executed in batches up to
    # the processes limit, queueing them all at once can be taxing on system
    # memory if buckets are very large. Adjust max_results as needed for your
    # system environment, or set it to None if you are sure the bucket is not
    # too large to hold in memory easily.
    # max_results=1000

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    os.makedirs(destination_directory, exist_ok=True)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=max_results)]

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))