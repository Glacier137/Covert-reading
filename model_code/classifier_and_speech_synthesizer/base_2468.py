import os
import math
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import scipy.io as scio
import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Softmax

from paths import DATA_ROOT

if float(str(tf.__version__.split('.')[1]))>=5:
    from tensorflow.keras.optimizers import SGD
else:
    from keras.optimizers import SGD

TF_GPU=tf.config.list_physical_devices('GPU')
print('tf version: '+ str(tf.__version__)+ '; \ntf GPU: '+str(TF_GPU))
physical_devices = tf.config.list_physical_devices()

# Print the available device names.
for device in physical_devices:
    print(device.name)

def sample_percentages(input_group, output_group,num,i,percen, seed=2):
    cut_input=[]
    cut_output=[]
    if num in [45, 47 , 50 , 54]:
        spec_power=50
    elif num in  [ 48 , 78]:
        spec_power=40
    else:
        spec_power=30
    random.seed(seed)
    nums = random.sample(range(0, spec_power), int(spec_power*percen))
    
    for syll in range(i):
        cut_input.extend([input_group[syll*spec_power:(syll+1)*spec_power][rd] for rd in nums ])
        cut_output.extend([output_group[syll*spec_power:(syll+1)*spec_power][rd] for rd in nums ])
    
    cut_input,cut_output=np.array(cut_input),np.array(cut_output)
        
    return cut_input,cut_output
    

def load_data(num,covert_or_overt,elecs,percenta,band):
    data_root = DATA_ROOT
    if band == 'b1':
        ori_data=scio.loadmat(os.path.join(data_root, "HSblockdata", "HS"+str(num)+"_Block_overt_covert_12_24_zscore_100Hz.mat"))
    if band == 'hg':
        ori_data=scio.loadmat(os.path.join(data_root, "HSblockdata", "HS"+str(num)+"_Block_overt_covert_70_150_zscore_100Hz.mat"))
    if num == 71:
        keys = [x for x in list(ori_data.keys()) if covert_or_overt in x and 'ma' not in x and 'na' not in x ]
    else:
        keys = [x for x in list(ori_data.keys()) if covert_or_overt in x]
    print(num,keys)
    num_all=len(keys)
    ECoG={}
    all_erps=[]
    labels=[]
    
    for i in range(num_all):
        if num in [45, 47 , 50 , 54]:
            ECoG[i]=ori_data[keys[i]][()][0:50,elecs,100:300]
        else:
            if num in [ 48 , 78] :
                # print(keys[i])
                ECoG[i]=ori_data[keys[i]][()][0:40,elecs,100:300]
            if num == 76 :
                ECoG[i]=ori_data[keys[i]][()][0:30,elecs,100:300]
                
            if num == 71 and covert_or_overt == 'ECoG_covert':
                ECoG[0]=ori_data['ECoG_covert_ba'][()][0:30,elecs,100:300]
                ECoG[1]=np.concatenate((ori_data['ECoG_covert_da'][()][0:15,elecs,100:300],
                                            ori_data['ECoG_covert_da'][()][0:15,elecs,100:300],),axis=0)
                ECoG[2]=ori_data['ECoG_covert_ga'][()][0:30,elecs,100:300]
                ECoG[3]=np.concatenate((ori_data['ECoG_covert_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_covert_pa'][()][10:20,elecs,100:300],
                                            ori_data['ECoG_covert_pa'][()][0:10,elecs,100:300]),axis=0)
                ECoG[4]=ori_data['ECoG_covert_ta'][()][0:30,elecs,100:300]
                ECoG[5]=ori_data['ECoG_covert_ka'][()][0:30,elecs,100:300]

                ECoG[6]=np.concatenate((ori_data['ECoG_covert_sa'][()][0:15,elecs,100:300],
                                            ori_data['ECoG_covert_sa'][()][0:15,elecs,100:300]),axis=0)

                ECoG[7]=np.concatenate((ori_data['ECoG_covert_sha'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_covert_sha'][()][10:20,elecs,100:300],
                                            ori_data['ECoG_covert_sha'][()][0:10,elecs,100:300]),axis=0)
            if num == 71 and covert_or_overt == 'ECoG_overt':    
                ECoG[0]=ori_data['ECoG_overt_ba'][()][0:30,elecs,100:300]
                ECoG[1]=np.concatenate((ori_data['ECoG_overt_da'][()][0:15,elecs,100:300],
                                            ori_data['ECoG_overt_da'][()][0:15,elecs,100:300],),axis=0)
                ECoG[2]=ori_data['ECoG_overt_ga'][()][0:30,elecs,100:300]
                ECoG[3]=np.concatenate((ori_data['ECoG_overt_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_overt_pa'][()][10:20,elecs,100:300],
                                            ori_data['ECoG_overt_pa'][()][0:10,elecs,100:300]),axis=0)
                ECoG[4]=ori_data['ECoG_overt_ta'][()][0:30,elecs,100:300]
                ECoG[5]=ori_data['ECoG_overt_ka'][()][0:30,elecs,100:300]

                ECoG[6]=np.concatenate((ori_data['ECoG_overt_sa'][()][0:15,elecs,100:300],
                                            ori_data['ECoG_overt_sa'][()][0:15,elecs,100:300]),axis=0)

                ECoG[7]=np.concatenate((ori_data['ECoG_overt_sha'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_overt_sha'][()][10:20,elecs,100:300],
                                            ori_data['ECoG_overt_sha'][()][0:10,elecs,100:300]),axis=0)
                
            if num == 73 and covert_or_overt == 'ECoG_covert':
                ECoG[0]=ori_data['ECoG_covert_ba'][()][0:30,elecs,100:300]
                ECoG[1]=ori_data['ECoG_covert_da'][()][0:30,elecs,100:300]
                ECoG[2]=ori_data['ECoG_covert_ga'][()][0:30,elecs,100:300]
                ECoG[3]=np.concatenate((ori_data['ECoG_covert_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_covert_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_covert_pa'][()][0:10,elecs,100:300]),axis=0)
                ECoG[4]=ori_data['ECoG_covert_ta'][()][0:30,elecs,100:300]
                ECoG[5]=np.concatenate((ori_data['ECoG_covert_ka'][()][0:20,elecs,100:300],
                                            ori_data['ECoG_covert_ka'][()][0:10,elecs,100:300]),axis=0)
                ECoG[6]=ori_data['ECoG_covert_sa'][()][0:30,elecs,100:300]
                ECoG[7]=ori_data['ECoG_covert_sha'][()][0:30,elecs,100:300]
            if num == 73 and covert_or_overt == 'ECoG_overt':
                ECoG[0]=ori_data['ECoG_overt_ba'][()][0:30,elecs,100:300]
                ECoG[1]=ori_data['ECoG_overt_da'][()][0:30,elecs,100:300]
                ECoG[2]=ori_data['ECoG_overt_ga'][()][0:30,elecs,100:300]
                ECoG[3]=np.concatenate((ori_data['ECoG_overt_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_overt_pa'][()][0:10,elecs,100:300],
                                            ori_data['ECoG_overt_pa'][()][0:10,elecs,100:300]),axis=0)
                ECoG[4]=ori_data['ECoG_overt_ta'][()][0:30,elecs,100:300]
                ECoG[5]=np.concatenate((ori_data['ECoG_overt_ka'][()][0:20,elecs,100:300],
                                            ori_data['ECoG_overt_ka'][()][0:10,elecs,100:300]),axis=0)
                ECoG[6]=ori_data['ECoG_overt_sa'][()][0:30,elecs,100:300]
                ECoG[7]=ori_data['ECoG_overt_sha'][()][0:30,elecs,100:300]
    input_shape=(len(elecs),ECoG[0].shape[2],1)
    for i in range(num_all):
        labels.append(i*np.ones(ECoG[i].shape[0]))
        all_erps.append(ECoG[i])
    all_erps = np.concatenate(all_erps, axis=0)
    labels = np.hstack(labels)
    print('HS=',num,all_erps.shape,len(labels))
    # CV_sp = np.concatenate([ECoG[i] for i in range(num_all)], axis=0)
    CV_sp=all_erps 
    hot=np.eye(num_all)
    all_hot=[]
    mix_TV={}
    for i in range(num_all):
        mix_TV[i]=np.tile((hot[i]),(ECoG[i].shape[0],1))
        # print(mix_TV[i].shape)
        all_hot.append(mix_TV[i])
    all_hot = np.concatenate(all_hot, axis=0)
    mix_sp=all_hot
    # mix_sp = np.array([np.argmax(i).tolist() for i in mix_sp])
    # The line above converts class-probability outputs to class indices if needed.
    return CV_sp, mix_sp,num_all,input_shape

def get_acc(a,b):
    acc=0
    for i in range(len(a)):
        if a[i]==b[i]:
            acc+=1
    real_acc=acc/len(a)
    return real_acc

def turn_percenta(percenta):
    writelist=["20%","40%","60%","80%","100%"]
    numlist=[0.2,0.4,0.6,0.8,1.0]
    return(numlist[writelist.index(percenta)])


def randC(num,mix_sp,i,num_all):
    if mix_sp.shape[0]/mix_sp.shape[1]<10:
        Ran_random_num=np.floor(mix_sp.shape[0]/10)
        Rand = random.sample(range(0, int(mix_sp.shape[0])), int(Ran_random_num))
    elif mix_sp.shape[0]/mix_sp.shape[1]>=10:
        Rand=[]
        basenum=int(mix_sp.shape[0]/mix_sp.shape[1])
        spec_power=basenum/10
        test_num_start=i*spec_power
        for j in range(num_all):
            start_num=int(np.floor(test_num_start+j*spec_power*10))
            end_num=int(math.ceil(test_num_start+j*spec_power*10+spec_power))
            Ran_temp=np.arange(start_num,end_num,1)
            Rand.append(Ran_temp)
    return Rand

def groupdevided(Allgroup,Ran,num_all):
    Allgroup_train=Allgroup.tolist()
    Allgroup_test=[]
    for i in range(num_all):
        for x in Ran[i]:
            a=np.where(Ran[i]==x)
            a=a[0][0]+i*num_all
            Allgroup_train.remove(Allgroup_train[x-a])
            Allgroup_test.append(Allgroup[x])  
    Allgroup_train=np.asarray(Allgroup_train)
    Allgroup_test=np.asarray(Allgroup_test)      
    return Allgroup_train, Allgroup_test

def get_acc(a,b):
    acc=0
    for i in range(len(a)):
        if a[i]==b[i]:
            acc+=1
    real_acc=acc/len(a)
    return real_acc

def accuR(CNNres,num, num_all):
    if num in [45, 47 , 50 , 54]:
        spec_power=5
    elif num in  [ 48 , 78]:
        spec_power=4
    else:
        spec_power=3
    results = {f"R{i}": 0 for i in range(num_all)}  # Initialize counters R0 through R(num_all - 1).

    for i in range(num_all):  # Iterate over the class groups.
        for j in range(i * spec_power, (i + 1) * spec_power):  # Iterate over the samples assigned to this class.
            if np.argmax(CNNres[0][j]) == i:  # Count predictions that match the current class.
                results[f"R{i}"] += 1

    accra = sum(results[f"R{i}"] for i in range(num_all)) / (num_all * spec_power)  # Calculate overall accuracy.
    return accra, results



def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides = strides, padding = 'same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def residual_a_or_b_or_c_or_d(input_x, filters, flag):
    if flag == "a":
        x = Conv_BN_Relu(filters, (3,3), 1, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        y = Add()([input_x, x])
        return y
    elif flag == "b":
        x = Conv_BN_Relu(filters, (3,3), 2, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        input_x = Conv_BN_Relu(filters, (1,1), 2, input_x)
        y = Add()([input_x, x])
        return y
    elif flag == "c":
        x = Conv_BN_Relu(filters, (1,1), 1, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1,1), 1, x)
        y = Add()([x, input_x])
        return y
    elif flag == "d":
        x = Conv_BN_Relu(filters, (1,1), 2, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1,1), 1, x)
        input_x = Conv_BN_Relu(filters * 4, (1,1), 2, input_x)
        y = Add()([x, input_x])
        return y

# ResNet-34 classifier.
def Vowel_model(
    CV_recon_train,
    output_train,
    CV_recon_test,
    epoch_num,
    verbose_set,
    Dense_num,
    input_shape,
    count,
    batch_size=32,
):
    
    inputs_CV= Input(shape=input_shape,name="inputs_CV")
    conv1 = Conv_BN_Relu(64, (7, 7), 1, inputs_CV)
    conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)
    x = conv1_Maxpooling
    
    # Residual feature-extraction layers.
    filters = 64
    num_residuals = [3, 4, 6, 3]
    for i, num_residual in enumerate(num_residuals):
        for j in range(num_residual):
            if j == 0:
                x = residual_a_or_b_or_c_or_d(x, filters, 'b')
                # x = residual_a_or_b_or_c_or_d(x, filters, 'd')
            else:
                x = residual_a_or_b_or_c_or_d(x, filters, 'a')
                # x = residual_a_or_b_or_c_or_d(x, filters, 'c')
        filters = filters * 2

    # Classification head.
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(Dense_num)(x)
    x = Dropout(0.4)(x)
    output_01 = Softmax(axis=-1,name="output_01")(x)

    model = Model(inputs=[inputs_CV],outputs=[output_01])

    if count==0:
        model.summary()

    losses={'output_01':'categorical_crossentropy'}


    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=sgd, 
                loss=losses, 
                metrics=['accuracy'], 
                loss_weights=None, 
                weighted_metrics=None)

    
    history = model.fit(
        CV_recon_train,       # Input
        output_train,         # Output (tensor instead of dict)
        batch_size=batch_size,
        epochs=epoch_num,
        verbose=verbose_set
    )

    
    yhat=model.predict({'inputs_CV':CV_recon_test},
                verbose=verbose_set)
    
    return model, history, yhat
