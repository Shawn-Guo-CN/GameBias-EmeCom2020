from recon_game import SymbolicReconGame

game = SymbolicReconGame(training_log='log/recon_train.txt')
game.train(10000)