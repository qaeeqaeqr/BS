# 毕业设计： 多智能体均值-方差线性组合最小强化学习算法预研

> author: 王钰

### Introduction

本项目主要将CTD对于回报的均值和方差的预测扩展到了多智能体的情况。

### Environment

##### 1 Code environment
本项目在windows下运行，python=3.9，
依赖库见[requirements.txt](requirements.txt)。

注意，在windows下，由于multi-agent-ale-py包的限制，
pettingzoo的atari环境均不可用。

##### 2 MARL environment

计划使用pettingzoo的以下环境：

* from pettingzoo.sisl import pursuit_v4
* from pettingzoo.butterfly import cooperative_pong_v5
* from pettingzoo.classic import connect_four_v3

这些环境详见pettingzoo官方文档[pettingzoo](https://pettingzoo.farama.org/)

### Project

[assets](./assets)中包含多智能体均值-方差线性组合最小算法的公式推导的tex文档。

[docs](./docs)中包含毕业设计的过程文档。

[configs](./configs)包含算法参数的配置。

[envs.py](./envs.py)中包含MARL的环境及其配置。

[test.py](./tests.py)中是一些测试代码，可忽略。




