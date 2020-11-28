In this small example, we cast VAE as a communication game between Sender (Encoder) and Receiver (Decoder).

This would be used as an example for the compositionality-as-disentanglement metrics.

In order to train on the entire dSprites dataset:
```bash
git clone https://github.com/Slowika/EGG_research;
cd EGG_research;
python3 -m venv venv;
source venv/bin/activate;
pip install wheel editdistance;
pip install --editable .;
python -m egg.zoo.dsprites_vae.train --lr=1e-3 --batch_size=128 --n_epochs=100 --vocab_size=6 --dataset=dsprites --subsample=1
```
where `vocab_size` sets the dimensionality of the latent representation.
