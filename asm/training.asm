; (A) Initialize the 3 Principal ANNs with default dropout
OP_NEUR CONFIG_ANN 0 FINALIZE 0.2
OP_NEUR CONFIG_ANN 1 FINALIZE 0.2
OP_NEUR CONFIG_ANN 2 FINALIZE 0.2

; (B) Tune hyperparameters via a genetic algorithm
OP_NEUR TUNE_GA 0 5 8
OP_NEUR TUNE_GA 1 5 8
OP_NEUR TUNE_GA 2 5 8

; (C) Train Principal ANNs using the tuned parameters
OP_NEUR TRAIN_ANN 0 40 0.005 16
OP_NEUR TRAIN_ANN 1 40 0.005 16
OP_NEUR TRAIN_ANN 2 40 0.005 16

; (D) Save all weights
OP_NEUR SAVE_ALL trained_weights

; Training complete – display a green indicator to confirm success
HALT
