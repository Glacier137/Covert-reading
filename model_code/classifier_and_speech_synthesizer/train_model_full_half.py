"""Historical full-versus-half electrode classifier variant."""
import argparse
import json
import numpy as np
import time
import gc
from tensorflow.keras import backend as K
from base_full_half import load_data, Vowel_model
from paths import ELECTRODE_LIST_PATH, RESULTS_ROOT as MODEL_RESULTS_ROOT
from sklearn.model_selection import KFold

def get_acc(a,b):
    acc=0
    for i in range(len(a)):
        if a[i]==b[i]:
            acc+=1
    real_acc=acc/len(a)
    return real_acc
RESULT_ROOT = MODEL_RESULTS_ROOT / 'full_half'
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
with ELECTRODE_LIST_PATH.open(encoding='utf-8') as f:
    electrode_lists = json.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--HS', type=int, required=True)
parser.add_argument('--condition', type=str, required=True)
parser.add_argument('--percent', type=str, required=True)
parser.add_argument('--elec', type=str, required=True)
parser.add_argument('--band', type=str, required=True)
args = parser.parse_args()

num = args.HS
covert_or_overt = args.condition
percenta = args.percent
elec=args.elec
band=args.band

save_path = RESULT_ROOT / f'{band}_{elec}_{num}_{covert_or_overt}_acc_251013.npy'
try:
    all_acc = np.load(save_path, allow_pickle=True).item()
except FileNotFoundError:
    all_acc = {}

key = f'{num}_{covert_or_overt}_{percenta}'
if key in all_acc:
    print(f"Skipping {key}, already processed.")
    exit()
    
    
if covert_or_overt == 'ECoG_overt':
    ele_state='SA'
elif covert_or_overt == 'ECoG_covert':
    ele_state='SI'    

if elec=='full_elecs':
    elecs=electrode_lists['full_half']['full_elecs']
elif elec=='half_elecs':
    elecs=electrode_lists['full_half']['half_elecs']


CV_sp, mix_sp, num_all, input_shape = load_data(num, covert_or_overt, elecs, percenta,band)


print(f'HS={num}, {covert_or_overt}, {CV_sp.shape}, {mix_sp.shape}, {percenta}')
print(f'Input shape: {input_shape}')

acR = {}
start_time = time.time()

kf = KFold(n_splits=10, shuffle=True, random_state=2)
acc_record=[]
for train_idx, test_idx in kf.split(CV_sp):
    
    CV_recon_train,CV_recon_test= CV_sp[train_idx],CV_sp[test_idx]
    Vowelrc_train, Vowelrc_test= mix_sp[train_idx],mix_sp[test_idx]
    output_train = Vowelrc_train

    epoch_num = 100
    verbose_set = 0

    model, history, *temp = Vowel_model(
        CV_recon_train, output_train, CV_recon_test,
        epoch_num, verbose_set, num_all, input_shape, count=1
    )

    # Optional accuracy calculation for class-label outputs.
    # acR[i] = accuR(temp, num, num_all)
    # print(f'Run {i}: {acR[i]}')
    
    a=[np.argmax(i).tolist() for i in temp[0]]
    b=[np.argmax(i).tolist() for i in Vowelrc_test]
    print(get_acc(a,b))
    acR.append( get_acc(a,b))
    print(f'Run {i}: {acR[i]}')

    del model, history, temp, CV_recon_train, CV_recon_test
    del Vowelrc_train, Vowelrc_test, output_train, Ran
    K.clear_session()
    gc.collect()

end_time = time.time()
meanAccuR = np.mean([acR[i][0] for i in range(10)])
print(f'HS={num}, mean accuracy = {meanAccuR:.3f}')
print(f'HS={num}, training time = {end_time - start_time:.3f} seconds')

all_acc[key] = [acR[i][0] for i in range(10)]
np.save(save_path, all_acc)

del CV_sp, mix_sp, num_all, input_shape, elecs_b1,elecs_hg, acR
K.clear_session()
gc.collect()
