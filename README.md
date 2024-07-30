# KGC-GCMIPA
**GCMIPA is under review in TKDE 2023.**  
The related datasets can be downloaded in [NELL-995](https://drive.google.com/file/d/18MnATMH7EYh0qcoCVS0V6QFQUfiloSJN/view?usp=sharing) [FB15k-237](https://drive.google.com/file/d/1Pj_aSIHKvWyzUSfzDLGFNf3yTCe1znCX/view?usp=sharing), then, put them into "/datasets".  

To reproduce the results, the hyperparameters are set in `model.py`.
```
python model.py NELL-995 concept_organizationheadquarteredincity 10 10
```

# Baselines
[RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
[HAKE](https://github.com/MIRALab-USTC/KGE-HAKE)
[PRA](https://github.com/David-Lee-1990/Path-ranking-algorithm)
[DeepPath](https://github.com/xwhan/DeepPath)
[MINERVA](https://github.com/shehzaadzd/MINERVA)
[CURL](https://github.com/RutgersDM/DKGR/tree/master)
[GRAIL](https://github.com/kkteru/grail)
[NBFNet](https://github.com/DeepGraphLearning/NBFNet)
[PathCon](https://github.com/hwwang55/PathCon)
[SQUIRE](https://github.com/bys0318/SQUIRE)

If you find this paper useful, please cite this paper:
```
@article{DBLP:journals/tkde/ZhuoWZW24,
  author       = {Xingrui Zhuo and
                  Gongqing Wu and
                  Zan Zhang and
                  Xindong Wu},
  title        = {Geometric-Contextual Mutual Infomax Path Aggregation for Relation
                  Reasoning on Knowledge Graph},
  journal      = {{IEEE} Trans. Knowl. Data Eng.},
  volume       = {36},
  number       = {7},
  pages        = {3076--3090},
  year         = {2024},
  url          = {https://doi.org/10.1109/TKDE.2024.3360258},
  doi          = {10.1109/TKDE.2024.3360258},
  timestamp    = {Tue, 18 Jun 2024 09:25:13 +0200},
  biburl       = {https://dblp.org/rec/journals/tkde/ZhuoWZW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
