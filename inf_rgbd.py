'最终融合'
import pandas as pd
import numpy as np
def softmax(x, dim=1):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)

slowonly_RGB_test93 = np.load("./inference_testphase/slowonly_addvalset_testset93_epoch62_RGB.npy")
slowonly_depth_test93 = np.load("./inference_testphase/slowonly_addvalset_testset93_epoch66_depth.npy")

'不加伪标签的模型'
slowonly_RGB = np.load("./inference_testphase/slowonly_addvalset_lr0.01_cropratio0.08_epoch71.npy")
slowonly_depth = np.load("./inference_testphase/slowonly_addvalset_epoch68_depth.npy")

slowonly_detect_RGB = np.load("./inference_testphase/slowonly_addvalset_detect_RGB_epoch77.npy")
slowonly_detect_depth = np.load("./inference_testphase/slowonly_addvalset_detect_depth_epoch53.npy")

slowonly_RGB_input256_inference288 = np.load("./inference_testphase/slowonly_addvelset_RGB_input256_inference288_epoch80.npy")
skeleton = np.load("./inference_testphase/skeleton_acc0.58.npy")


TSM_RGB_flow = np.load("./inference_testphase/TSM_addvalset_RGB_flow_Epoch35.npy")
TSM_depth_flow = np.load("./inference_testphase/TSM_addvalset_depth_flow_Epoch35.npy")
TSM_RGB_finetune = np.load("./inference_testphase/TSM_addvalset_RGB_finetune_Epoch20.npy")

'加伪标签的模型'
slowonly_RGB_pse = np.load("./inference_testphase/slowonly_addvelset_addtestsetv1_lr_0.01_cropratio0.08_epoch87.npy")
slowonly_depth_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_depth_epoch81.npy")
slowonly_detect_RGB_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_detect_RGB_epoch92.npy")
slowonly_detect_depth_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_detect_depth_epoch60.npy")

TSM_RGB_flow_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_RGBflow_finetune_Epoch10.npy")
TSM_RGB_finetune_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_RGB_finetune_Epoch10.npy")
TSM_depth_flow_pse = np.load("./inference_testphase/TSM_addvalset_addtestsetv1_depth_flow_Epoch10.npy")

slowonly_RGB_input256_inference288_pse = np.load("./inference_testphase/slowonly_addvalset_addtestsetv1_input256_inference288_epoch98.npy")
slowfast_RGB_detect_pse = np.load("./inference_testphase/slowfast_test95_detect_RGB_epoch75.npy")
slowfast_depth_detect_pse = np.load("./inference_testphase/slowfast_test95_detect_depth_epoch74.npy")

print(np.array(slowonly_RGB_input256_inference288_pse).shape)
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
# print(np.max(slowfast_RGB_256_nln, axis=1))
# print(slowfast_flowx_256_nln + slowonly_RGB_input256_inference288_pse)
# import pdb;pdb.set_trace()


            

corporate = 0*slowonly_RGB \
        + 0*slowonly_detect_RGB \
        + 0*slowonly_RGB_input256_inference288 \
        + 0*skeleton \
        + 0*TSM_RGB_flow \
        + 0*TSM_RGB_finetune \
                            \
        + 1 * slowfast_RGB \
        + 2.5 * slowfast_RGB_256 \
        + 1 * slowfast_RGB_256_nln \
        + 1 * slowfast_RGB_nln \
        + 2.5 * slowfast_seg_nln \
        + 1 * slowfast_flowx_256_nln \
        + 1 * slowfast_flowy_nln \
        + 1.1 * slowfast_depth 
        
        
        
        

corporate1 = 0*slowonly_RGB \
        + 0*slowonly_detect_RGB \
        + 0*slowonly_RGB_input256_inference288 \
        + 0*skeleton \
        + 0*TSM_RGB_flow \
        + 0*TSM_RGB_finetune \
        +  1.5 * slowonly_RGB_pse \
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

my_predictions = pd.read_csv("predictions.csv",header=None)
