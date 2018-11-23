# input params
GRAY = True                # read images as gray scale
INPUT_SIZE = [256, 256]
INPUT_CH = 1 if GRAY else 3 # set automatically by GRAY parameter


# architecture params
KERNELS_SIZE = [[5,5, INPUT_CH, 4], [3, 3, 4, 4]]  # HWIO - height, width, input_ch, output_ch
NLAYER_SIZES = [[64, 64],[64,64]]#, [15,15], [8,8], [4,4]]
WINDOW = [4, 1]
STRIDE = [4, 1]


# learning params
batch_size = 2
NUM_LAYERS = 2

LEARNING_RATE = 0.001
ITERATION = 1000000000
load_all_prev = 0
train_all_prev = 1

# tensorborad
SUMMARIES_DIR = './summaries'