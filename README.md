# GameBias

## 1. Installation

```bash
# start by installing Slowika/EGG_research 
git clone https://github.com/Slowika/EGG_research;
cd EGG_research;
python3 -m venv venv;
source venv/bin/activate;
pip install wheel editdistance;
pip install .;
# then install Shawn-Guo-CN/GameBias
cd ..;
git clone https://github.com/Shawn-Guo-CN/GameBias;
cd GameBias;
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
