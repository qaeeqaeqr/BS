# 毕业设计： 多智能体均值-方差线性组合最小强化学习算法预研

> author: 王钰

### Introduction

本项目主要将CTD对于回报的均值和方差的预测扩展到多智能体的情况。

### Environment

````
conda create -n ctd-marl python=3.9
conda activate ctd-marl
pip install -r requirements.txt
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
````


### Project

主要实现两个环境，分别在[food_collector_env](food_collector_env)和
[simple_high_variance_env](simple_high_variance_env)里；
这两个文件夹里也包含iql、ma-ctd-iql、vdn、ma-ctd-vdn的算法实现及绘图。

其余代码仍进一步开发中。

### todo

funhpc服务器：ssh -p 46480 root@ykvvhm6pxtisq8ajsnow.deepln.com
密码：X703IOrLrm3VyTGq3kpy0DdLjYuoeMdg

