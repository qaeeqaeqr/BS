# 毕业设计： 多智能体均值-方差线性组合最小强化学习算法预研

> author: 王钰

### Introduction

本项目主要将CTD对于回报的均值和方差的预测扩展到多智能体的情况。

### Environment

##### 1 Code environment
python=3.9，
依赖库见[requirements.txt](requirements.txt)。

注意，在windows下，由于multi-agent-ale-py包的限制，
pettingzoo的atari环境均不可用。

##### 2 MARL environment

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

1、本机算力不足

2、目前只考虑了pursuit的环境

