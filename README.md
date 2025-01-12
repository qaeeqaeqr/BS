# 毕业设计： 多智能体均值-方差线性组合最小强化学习算法预研

> author: 王钰

### Introduction

本项目主要将CTD对于回报的均值和方差的预测扩展到多智能体的情况。

### Environment

##### 1 Code environment

在linux环境下，令当前工作目录为本项目根文件夹，顺序运行以下指令即可：
````
conda create -n ctd-marl python=3.9
conda activate ctd-marl
pip install -r requirements.txt
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
````

##### 2 MARL environment

注意，在windows下，由于multi-agent-ale-py包的限制，
pettingzoo的atari环境均不可用。

计划使用pettingzoo的以下环境：

* from pettingzoo.sisl import pursuit_v4

环境详见pettingzoo官方文档
[pettingzoo](https://pettingzoo.farama.org/)。

### Project

[algorithms](./algorithms)中包含各种算法的实现。

[assets](./assets)中包含算法的公式推导的tex文档。

[configs](./configs)包含算法参数的配置。

[docs](./docs)中包含毕业设计的过程文档。

[models](./models)存放训练好的模型

[outputs](./outputs)存放程序输出，如训练reward变化折线图、
可视化结果等。

[train.py](./train.py)是训练脚本。

[visualize.py](./visualize.py)完成算法可视化。



### todo

0、加入learning rate decay

1、本机算力不足。预计用[服务器](https://www.funhpc.com/)训练

2、目前只考虑了pursuit的环境，而pursuit环境似乎过于复杂，
目前模型学习不到好的策略，故需要考虑简单的环境。

