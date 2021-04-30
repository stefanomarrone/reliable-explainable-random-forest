# reliable-explainable-random-forest

# wnl-tools
Water Network Language editor and analysis tools

This repository contains the source code and the Eclipse plugins of the Water Network Language (WNL) tools. 

## Motivation
The software here contained is related to a scientific research providing a modelling and analysis framework for the vulnerability evaluation of complex transport systems, i.e. networked systems in charge of transporting goods across a physical space as logistic networks, oil & gas networks, etc. In particular, the WNL language is built for the modeling and the evaluation of water supply networks. The approach is based on a traditional model-driven schema "model-translate-analyse":
* the modeling phase uses a DSML able to model both the physical stucture of a water supply network (node, pipes, tanks, reservoirs, etc.), the logical insfrastructure able to monitor such physical networks (sensors, communiction devices, etc.) and the attack (both to the physical and to the monitoring infrastructure);
* the translation phase generates a "low-level model" able to be analysed in a more efficient manner. In particular, the formalism of the Bayesian Networks has been chosen due to it ability to model directed acyclic graphs providing a well-founded framework for the reasoning under uncertainty;
* the analysis level is able to provide both a-priori analysis (which is the probability of having a successful attack), a-posteriori analysis (given this kind of attack, which is the probability of having a damage in the network) and a diagnosis analysis (since sensors have measured an abnormal activity, which is the most probable cause?).

## Content of the repository
The repository is structured in the following folders:
* *WNL language*: that is composed of both the WNL Ecore language and the textual notation.
* *WNL Editor*: graphical/textual editor able to produce WNL model conformant to the WNL Ecore language and textual notation (Eclipse project).
* *WNL Analyser*: back-end tool that translates the high level model into a Bayesian Network and calls the proper solver according to the specific analysis to provice  (Eclipse project).
* *Case study*: files related to the Jowitt and Xu case study.

## License
The software is licensed according to the GNU General Public License v3.0 (see License file).

## People
* Stefano Marrone - Università della Campania "Luigi Vanvitelli" (Italy)
* Ugo Gentile - CERN (Switzerland)

## Credits
This software is build upon the following software libraries. Without them, building this software would be harder:
* SableCC ver. 3.7 - http://sablecc.org/
* JavaBayes ver 0.346 - https://www.cs.cmu.edu/~javabayes/Home/ (modified without GUI)

## Bibliography
Please refer to the following paper to de
1. Gentile, U., Marrone, S., De Paola, F., Nardone, R., Mazzocca, N., Giugni, M.; Model-Based Water Quality Assurance in Ground and Surface Provisioning Systems; Proceedings - 2015 10th International Conference on P2P, Parallel, Grid, Cloud and Internet Computing, 3PGCIC 2015; 2016; 10.1109/3PGCIC.2015.97
1. Aquasystem: italian funded project for innovative techniques and practices for improving quality and managent of surface water supply networks (http://www.aquasystemproject.it/)
1. Optimal Valve Control in Water Distribution Networks
Paul W. Jowitt, Chengchao Xu; Optimal Valve Control in Water Distribution Networks; Journal of Water Resources Planning and Management; Vol. 116, Issue 4 (July 1990) https://doi.org/10.1061/(ASCE)0733-9496(1990)116:4(455)