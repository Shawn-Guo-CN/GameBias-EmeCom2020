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
