# GameBias

## 1. Installation

```bash
# 1. create a virtual environment for the repo
python3 -m venv venv;
source venv/bin/activate;
# 2. install the editdistance package for EGG_research module
pip install wheel editdistance;
# 3. install lib/EGG_research
cd lib/EGG_research;
pip install .
# 4. install other dependencies
cd ../../
pip install -r requirements.txt;
```

## 2. Run Experiment

```bash
# NOTE that the virtual environment created above should be activated first

cd bashes
# run the experiment on comparing the compositionality degree of emergent languages from different types of game
bash topo_simi_refer_recon.sh

# run the experiment on comparing the expressivity of emergent languages from different types of game
bash task_transfer_experiment.sh

# plot the results
cd ..
python analysis

```
## 3. Reference

If you find the code useful, please cite our [paper](https://arxiv.org/abs/2012.02875) presented at the [4th NeurIPS Workshop on Emergent Communication: Talking to Strangers: Zero-Shot Emergent Communication](https://sites.google.com/view/emecom2020/home):

```
@article{DBLP:journals/corr/abs-2012-02875,
  author    = {Shangmin Guo and
               Yi Ren and
               Agnieszka Slowik and
               Kory W. Mathewson},
  title     = {Inductive Bias and Language Expressivity in Emergent Communication},
  journal   = {CoRR},
  volume    = {abs/2012.02875},
  year      = {2020},
  url       = {https://arxiv.org/abs/2012.02875},
  eprinttype = {arXiv},
  eprint    = {2012.02875},
  timestamp = {Fri, 17 Dec 2021 11:39:29 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2012-02875.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

