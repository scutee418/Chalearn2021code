import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os

file_ = "work_dirs/slowonly_dict_train_useDataparallel/20210206_164149.log.json"
list = []
for line in open(file_,'r'):
    list.append(line)
lr_list = []
loss_list = []
top1_acc_list = []
cnt = 0
for i in list:
    if "mode" not in i:
        continue
    if int(''.join(filter(str.isdigit,i.split("epoch")[1].split("iter")[0]))) == cnt:
        continue
    lr_block = i.split("lr")[1].split("memory")[0]
    if "e" in lr_block:
        lr_ahead = ''.join(filter(str.isdigit,lr_block.split("e")[0]))
        lr_later = ''.join(filter(str.isdigit,lr_block.split("e")[1]))
        lr_value = lr_ahead + "e-" + lr_later
    else:
        lr_ahead = ''.join(filter(str.isdigit,lr_block.split(".")[0]))
        lr_later = ''.join(filter(str.isdigit,lr_block.split(".")[1]))
        lr_value = lr_ahead + "." + lr_later
    lr_list.append(float(lr_value))
    loss_block = i.split("loss_cls")[1].split("loss")[0]
    loss_ahead = ''.join(filter(str.isdigit,loss_block.split(".")[0]))
    loss_later = ''.join(filter(str.isdigit,loss_block.split(".")[1]))
    loss_value = loss_ahead + "." + loss_later
    loss_list.append(float(loss_value))
    if "top1_acc" in i:
        top1_acc_block = i.split("top1_acc")[1].split("top5_acc")[0]
        top1_acc_ahead = ''.join(filter(str.isdigit,top1_acc_block.split(".")[0]))
        top1_acc_later = ''.join(filter(str.isdigit,top1_acc_block.split(".")[1]))
        top1_acc_value = top1_acc_ahead + "." + top1_acc_later
        top1_acc_list.append(float(top1_acc_value))
    cnt += 1

save_dir = os.path.join("lr_analyze_figures",file_.split("/")[1])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.figure(1)
plt.plot(range(cnt),lr_list)
plt.savefig(os.path.join(save_dir,"lr.jpg"))
plt.figure(2)
plt.plot(range(cnt),loss_list)
ax = plt.gca()
x_major_locator=MultipleLocator(5)  
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator=MultipleLocator(0.5)  
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(-0.5,6)
plt.text(cnt, loss_list[-1], '%.4f' % loss_list[-1], ha='center', va= 'bottom',fontsize=9)
plt.savefig(os.path.join(save_dir,"loss.jpg"))
if top1_acc_list:
    plt.figure(3)
    plt.plot(range(cnt),top1_acc_list)
    ax = plt.gca()
    x_major_locator=MultipleLocator(5)  
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator=MultipleLocator(0.1)  
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0,1.1)
    plt.text(cnt, top1_acc_list[-1], '%.4f' % top1_acc_list[-1], ha='center', va= 'bottom',fontsize=9)
    plt.savefig(os.path.join(save_dir,"top1_acc.jpg"))
