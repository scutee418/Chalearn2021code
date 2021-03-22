'最终融合'
import pandas as pd
import numpy as np
def softmax(x, dim=1):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)



slowonly_RGB_pse = np.load("./inference_testphase/slowonly_addvelset_addtestsetv1_lr_0.01_cropratio0.08_epoch87.npy")
slowonly_depth_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_depth_epoch81.npy")
slowonly_detect_RGB_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_detect_RGB_epoch92.npy")
slowonly_detect_depth_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_detect_depth_epoch60.npy")

TSM_RGB_flow_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_RGBflow_finetune_Epoch10.npy")
TSM_RGB_finetune_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_RGB_finetune_Epoch10.npy")
TSM_depth_flow_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_depth_flow_Epoch10.npy")

slowonly_RGB_input256_inference288_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_input256_inference288_epoch98.npy")


# import pdb;pdb.set_trace()
import pickle
slowfast_RGB_nln = pickle.load(open('./inference_testphase/results_9543.pkl', 'rb'))[0].numpy() / 30
slowfast_seg_nln = pickle.load(open('./inference_testphase/results_seg.pkl', 'rb'))[0].numpy() / 30
slowfast_RGB_256_nln = pickle.load(open('./inference_testphase/results256_pse3.pkl', 'rb'))[0].numpy() / 90
slowfast_RGB = pickle.load(open('./inference_testphase/results.pkl', 'rb'))[0].numpy() / 30
slowfast_RGB_256 = pickle.load(open('./inference_testphase/results_y.pkl', 'rb'))[0].numpy() / 30
slowfast_flowy_nln = pickle.load(open('./inference_testphase/results_flowy.pkl', 'rb'))[0].numpy() / 30
slowfast_flowx_256_nln = pickle.load(open('./inference_testphase/results256_flowx.pkl', 'rb'))[0].numpy() / 90
slowfast_a_flowx = pickle.load(open('./inference_testphase/results_x.pkl', 'rb'))[0].numpy() / 30
slowfast_a_flowy = pickle.load(open('./inference_testphase/results_y.pkl', 'rb'))[0].numpy() / 30
slowfast_a_RGB = pickle.load(open('./inference_testphase/results_mg_test.pkl', 'rb'))[0].numpy() / 30
slowfast_a_RGB_256 = pickle.load(open('./inference_testphase/results_256_test.pkl', 'rb'))[0].numpy() / 90
slowfast_a_RGB_nln = pickle.load(open('./inference_testphase/results_nln_test.pkl', 'rb'))[0].numpy() / 30
slowfast_depth = pickle.load(open('./inference_testphase/results256_depth.pkl', 'rb'))[0].numpy() / 90


            

corporate = 1 * slowfast_RGB \
        + 2.5 * slowfast_RGB_256 \
        + 1 * slowfast_RGB_256_nln \
        + 1 * slowfast_RGB_nln \
        + 2.5 * slowfast_seg_nln \
        + 1 * slowfast_flowx_256_nln \
        + 1 * slowfast_flowy_nln \
        + 1.1 * slowfast_depth 
        
        
        
        

corporate1 = 1.5 * slowonly_RGB_pse \
        + 1.5 * slowonly_detect_RGB_pse \
        + 1 * slowonly_RGB_input256_inference288_pse \
        + 1 * TSM_RGB_flow_pse \
        + 1.5 * TSM_RGB_finetune_pse \
        + 0.3 * slowonly_depth_pse \
        + 1.3 * slowonly_detect_depth_pse \
        + 1.5 * TSM_depth_flow_pse
        




# corporate1 = corporate1
corporate = 1.5 * corporate  + 0.4 *corporate1 + 1.8 * slowonly_depth_pse  + 0.1 * slowfast_depth
predictions = pd.read_csv("./inference_testphase/predictions_testphase.csv",header=None)
line = []
for i in range(len(corporate)):
    pred = np.argsort(corporate[i])[-1]
    line.append([predictions.iloc[i,0],pred])
data = pd.DataFrame(data=line)
data.to_csv("predictions.csv",header=None,index=None,sep=",")
