OUTPUT_ROOT = "output"
# if OUTPUT_DIR="auto", osp.join(cfg.OUTPUT_ROOT, osp.splitext(args.config_file)[0].split("configs/")[1])
OUTPUT_DIR = "output"

EXP_NAME = ""

DEBUG = False
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
CUDNN_BENCHMARK = True
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
VIS_PERIOD = 0

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
INPUT = dict(
    # Whether the model needs RGB, YUV, HSV etc.
    FORMAT="BGR",
    MIN_SIZE_TRAIN=(480,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING="choice",
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=False,
    AUG_DEPTH=False,
    # color aug
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="ROI10D",
    COLOR_AUG_CODE="",
    COLOR_AUG_SYN_ONLY=False,
    ## bg images
    BG_TYPE="VOC_table",  # VOC_table | coco | VOC | SUN2012
    BG_IMGS_ROOT="datasets/VOCdevkit/VOC2012/",  # "datasets/coco/train2017/"
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.5,  # prob to change bg of real image
    # truncation fg (randomly replace some side of fg with bg during replace_bg)
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    ## bbox aug
    DZI_TYPE="uniform",  # uniform, truncnorm, none, roi10d
    DZI_PAD_SCALE=1.0,
    DZI_SCALE_RATIO=0.25,  # wh scale
    DZI_SHIFT_RATIO=0.25,  # center shift
    # smooth xyz map by median filter
    SMOOTH_XYZ=False,
)

# -----------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------
DATASETS = dict(
    TRAIN=(),
    TRAIN2=(),  # the second training dataset, useful for data balancing
    TRAIN2_RATIO=0.0,
    # List of the pre-computed proposal files for training, which must be consistent
    # with datasets listed in DATASETS.TRAIN.
    PROPOSAL_FILES_TRAIN=(),
    # Number of top scoring precomputed proposals to keep for training
    PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
    TEST=(),
    PROPOSAL_FILES_TEST=(),
    # Number of top scoring precomputed proposals to keep for test
    PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    DET_FILES_TEST=(),
    DET_TOPK_PER_OBJ=1,
    DET_THR=0.0,  # filter detections
    # NOTE: override if symmetric objects are different, used for custom evaluator
    # SYM_OBJS=["024_bowl", "036_wood_block", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"],  # ycbv
    # SYM_OBJS=["002_master_chef_can", "024_bowl", "025_mug", "036_wood_block", "040_large_marker", "051_large_clamp",
    #           "052_extra_large_clamp", "061_foam_brick"],  # ycbv_bop
    SYM_OBJS=["bowl", "cup", "eggbox", "glue"],
)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=4,
    ASPECT_RATIO_GROUPING=False,  # default True in detectron2
    # Default sampler for dataloader
    # Options: TrainingSampler, RepeatFactorTrainingSampler
    SAMPLER_TRAIN="TrainingSampler",
    # Repeat threshold for RepeatFactorTrainingSampler
    REPEAT_THRESHOLD=0.0,
    # If True, the dataloader will filter out images that have no associated
    # annotations at train time.
    FILTER_EMPTY_ANNOTATIONS=True,
    # NOTE: set to False if you want to see the image anyways
    FILTER_EMPTY_DETS=True,  # filter images with empty detections
    # filter out instances with visib_fract <= visib_thr at train time
    FILTER_VISIB_THR=0.0,
)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER = dict(
    IMS_PER_BATCH=6,
    TOTAL_EPOCHS=160,
    # NOTE: use string code to get cfg dict like mmdet
    # will ignore OPTIMIZER_NAME, BASE_LR, MOMENTUM, WEIGHT_DECAY
    OPTIMIZER_CFG=dict(type="RMSprop", lr=1e-4, momentum=0.0, weight_decay=0),
    #######
    GAMMA=0.1,
    BIAS_LR_FACTOR=1.0,
    LR_SCHEDULER_NAME="WarmupMultiStepLR",  # WarmupMultiStepLR | flat_and_anneal
    WARMUP_METHOD="linear",
    WARMUP_FACTOR=1.0 / 1000,
    WARMUP_ITERS=1000,
    ANNEAL_METHOD="step",
    ANNEAL_POINT=0.75,
    POLY_POWER=0.9,  # poly power
    REL_STEPS=(0.5, 0.75),
    # checkpoint
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
    # Enable automatic mixed precision for training
    # Note that this does not change model's inference behavior.
    # To use AMP in inference, run inference under autocast()
    AMP=dict(ENABLED=False),
)

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
TRAIN = dict(
    PRINT_FREQ=100,
    VERBOSE=False,
    VIS=False,
    # vis imgs in tensorboard
    VIS_IMG=False,
)
# ---------------------------------------------------------------------------- #
# Specific val options
# ---------------------------------------------------------------------------- #
VAL = dict(
    DATASET_NAME="lm",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    RESULTS_PATH="",
    TARGETS_FILENAME="lm_test_targets_bb8.json",
    ERROR_TYPES="ad,rete,re,te,proj",
    RENDERER_TYPE="cpp",  # cpp, python, egl, aae
    SPLIT="test",
    SPLIT_TYPE="bb8",
    N_TOP=1,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
    EVAL_CACHED=False,  # if the predicted poses have been saved
    SCORE_ONLY=False,  # if the errors have been calculated
    EVAL_PRINT_ONLY=False,  # if the scores/recalls have been saved
    EVAL_PRECISION=False,  # use precision or recall
    USE_BOP=False,  # whether to use bop toolkit
)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE="gt",  # gt | est
    # USE_PNP = False,  # use pnp or direct prediction
    # PNP_TYPE = "ransac_pnp",
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    AMP_TEST=False,
)
