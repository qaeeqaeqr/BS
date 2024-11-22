# Cascaded Temporal Difference (CTD)

> author: Yu Wang  
> last modified: 2024.7.23  
> github:  https://github.com/qaeeqaeqr/CTD

### Introduction

This is the python implementation for paper
"CTD: Cascaded Temporal Difference Learning for
the Mean-Standard Deviation Shortest Path Problem"

This project implement the CTD algorithm on
"Sioux Falls network". The "Sioux Falls network" 
used in this project is obtained from 
[https://github.com/bstabler/TransportationNetworks](https://github.com/bstabler/TransportationNetworks).

### Project Structure

CTD algorithm is implemented in script [CTD.py](CTD.py)

**note** that "get_reward" function may need to be rewriten
when handling different networks.

CTD arguments are configurated in script [argparser.py](argparser.py)

Sioux Falls network data is recorded from the orginal
CTD paper, and located in [Networks_data](Networks_data)

Scripts for loading network data are located in 
[Networks](Networks)

