## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation 在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [参考资料 Reference](#Reference)

## Top News
**`2024-01`**:**创建仓库，支持简单预测与NP实现的简单Attention。**   

### 所需环境
torch==1.7.1以上

### 文件下载  
训练所需的pth可以在百度网盘下载。       
链接: https://pan.baidu.com/s/1b9Nt-UuqOJfhbhJYVyrK0g     
提取码: mfnc     

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data
2. 运行predict_generate.py。 
3. 运行predict_itm.py.
4. predict_features.py  

### Reference
https://github.com/salesforce/BLIP  