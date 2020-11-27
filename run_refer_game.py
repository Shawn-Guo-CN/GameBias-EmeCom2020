from refer_game import SymbolicReferGame

game = SymbolicReferGame(training_log='log/refer_train.txt')
game.train(10000)