import os
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from pprint import pprint

def getDate(file_name):
    date, time = file_name.split("_", 2)[:2]
    date_n_time = date.split('.') + time.split('.')
    date_n_time_int = [int(x) for x in date_n_time]
    return(datetime(*date_n_time_int))

if __name__ == "__main__":
    experiments_folder = "./_experiments"
    folders_with_dates = [(getDate(name), name) for name in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder, name))]

    folders_with_dates.sort(key=lambda r: r[0])

    last_folder = folders_with_dates[-1][1]

    all_sep_results = [(json.load(open(f"{experiments_folder}/{last_folder}/{filename}"))) for filename in os.listdir(f"{experiments_folder}/{last_folder}") if filename.startswith("_per_instance_results_")]

    total_result = defaultdict(list)

    for res in all_sep_results:
        for key, value in res.items():
                total_result[key].append(value)

    total_result["mean_place_of_shap_set"] = np.nanmean(np.array(total_result['result_shap_places']))
    total_result["mean_difference_with_best"] = np.nanmean(np.array(total_result['result_differences']))
    total_result["mean_gains_in_time"] = np.nanmean(np.array(total_result['gains_in_time']))

    pprint(total_result, open(f'{experiments_folder}/{last_folder}/_interrupted_results.txt', 'w'))