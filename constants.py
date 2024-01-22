# hyperparameters (Do not change GPT_VOCAB_SIZE if using GPT-2 embeddings. Otherwise, figure out the optimal hyperparameters.)
BATCH_SIZE = 0 # How many independent sequences will we process in parallel?
BLOCK_SIZE = 0 # What is the maximum context length for predictions?
NUM_EPOCHS = 0 # How many times are we running through the dataset?
EVAL_INTERVAL = 0 # At what intervals are we evaulating the model and saving its state?
LEARNING_RATE = 0 # Experiment to find out what LR works best. Keep in mind that GPT is a very complicated model and thus, 
EVAL_ITERS = 0
N_EMBD = 0
N_HEAD = 0
N_LAYER = 0
DROPOUT = 0.0 # Probability of a neuron being deactivated during training
GPT_VOCAB_SIZE = 50257 # Vocab size of GPT-2 tiktoken-imported embeddings