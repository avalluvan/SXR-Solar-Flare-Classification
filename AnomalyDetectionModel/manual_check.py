import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.special import erf
import pandas as pd
from tqdm import tqdm

plt.ion()
plt.rcParams['figure.facecolor'] = 'white'

def EFP(time: np.int32 | npt.NDArray[np.int32], amplitude: np.float64, mu: np.float64, sigma: np.float64, tau: np.float64) -> np.float64 | npt.NDArray[np.float64]:
    Aprime = amplitude * sigma / (tau * np.sqrt(2 / np.pi))
    y1 = -(time - mu) / tau + sigma**2 / (2 * tau**2)
    y2 = (time - mu) / (np.sqrt(2) * sigma) - sigma / (np.sqrt(2) * tau)
    return Aprime * np.exp(y1) * (1 + erf(y2))

class_info = pd.read_csv('allflares_preprocessed.csv')

try:
    process = pd.read_csv('allflares_processed.csv')
except FileNotFoundError:
    process = pd.DataFrame(columns=class_info.columns)

try:
    ignore = pd.read_csv('ignore.csv')
except FileNotFoundError:
    ignore = pd.DataFrame(columns=class_info.columns)

for i, iden in enumerate(tqdm(np.array(class_info['identifier']))):
    if class_info['flare_type'][i] != 'A' and class_info['flare_type'][i] != 'B':
        continue

    if bool(class_info['multi_flare_region_flag'][i]): # ignore multi flare regions
        continue

    if class_info['rsquared'][i] < 0.5: # only consider good fits to eliminate incorrect validation dataset
        continue

    try:
        tod = pd.read_csv('flares/' + iden + '.csv')
    except FileNotFoundError:
        continue

    if np.max(np.diff(tod['Time'])) > 10 or np.max(np.diff(tod['Time'])) < 10:
        continue

    if iden in np.array(process['identifier']) or iden in np.array(ignore['identifier']):
        continue

    yfit = EFP(tod['Time'], class_info['amplitude'][i], class_info['mu'][i], class_info['sigma'][i], class_info['tau'][i])

    plt.scatter(tod['Time'], tod['Counts'])
    plt.plot(tod['Time'], yfit, c='r') 

    plt.title(iden + ': ' + class_info['flare_type'][i])

    plt.show()

    if int(input(f'{iden}: Single flare & correct type? [1 = yes, 0 = no] ')) == 1:
        process = pd.concat([process, class_info.iloc[[i]]], ignore_index=True)
        process.to_csv('allflares_processed.csv', index=False)
    else:
        ignore = pd.concat([ignore, class_info.iloc[[i]]], ignore_index=True)
        ignore.to_csv('ignore.csv', index=False)

    plt.clf()