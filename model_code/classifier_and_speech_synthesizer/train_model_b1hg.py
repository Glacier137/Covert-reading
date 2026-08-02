"""Historical combined beta1/high-gamma classifier variant."""
import argparse
import json
import numpy as np
import time
import gc
from tensorflow.keras import backend as K
from base_b1hg import load_data, Vowel_model
from paths import ELECTRODE_LIST_PATH, RESULTS_ROOT as RESULT_ROOT
from sklearn.model_selection import KFold

def get_acc(a,b):
    acc=0
    for i in range(len(a)):
        if a[i]==b[i]:
            acc+=1
    real_acc=acc/len(a)
    return real_acc
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
with ELECTRODE_LIST_PATH.open(encoding='utf-8') as f:
    electrode_lists = json.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--HS', type=int, required=True)
parser.add_argument('--condition', type=str, required=True)
parser.add_argument('--percent', type=str, required=True)
args = parser.parse_args()

num = args.HS
covert_or_overt = args.condition
percenta = args.percent

save_path = RESULT_ROOT / 'all_acc_b1hg_251010.npy'
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


elecs_b1=electrode_lists['classifier'][ele_state]['b1'][str(num)]
CV_sp_b1, mix_sp, num_all, input_shape_b1 = load_data(num, covert_or_overt, elecs_b1, percenta,'b1')
elecs_hg=electrode_lists['classifier'][ele_state]['hg'][str(num)]
CV_sp_hg, mix_sp, num_all, input_shape_hg = load_data(num, covert_or_overt, elecs_hg, percenta,'hg')
CV_sp=np.concatenate((CV_sp_b1,CV_sp_hg),axis=1)
input_shape=(CV_sp.shape[1],CV_sp.shape[2],1)

print(f'HS={num}, {covert_or_overt}, {CV_sp.shape}, {mix_sp.shape}, {percenta}')
print(f'Input shape: {input_shape}')

acR = {}
start_time = time.time()

# Optional evaluation when the output is represented as class labels.
# for i in range(10):
#     Ran,randindex = randC(num, mix_sp,i, num_all)
#     #print(Ran)
    
#     CV_recon_train, CV_recon_test = groupdevided(CV_sp.copy(), Ran, num_all,randindex)
#     Vowelrc_train, Vowelrc_test = groupdevided(mix_sp.copy(), Ran, num_all,randindex)
#     output_train = Vowelrc_train


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
        epoch_num, verbose_set, num_all, input_shape, count
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
