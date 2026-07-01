import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Replace with correct path as per you device
CELEB_PATH = r'D:\Projects\demo\dataset\CelebDF'
WILD_PATH  = r'D:\Projects\demo\dataset\WildDeepfake'

LOG_FILE          = 'training_log.csv'
WEIGHTS_STAGE2    = 'weights_stage2_fusion.pth'
WEIGHTS_STAGE3    = 'weights_stage3_meta.pth'
WEIGHTS_FINAL     = 'weights_final.pth'

BATCH_SIZE    = 8          
VAL_SPLIT     = 0.1
NUM_WORKERS   = 0          

LR_STAGE2 = 1e-3           
LR_STAGE3 = 5e-4           
LR_STAGE4 = 1e-5           
EPOCHS_STAGE2 = 10
EPOCHS_STAGE3 = 10
EPOCHS_STAGE4 = 10

WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

XCEPTION_DIM  = 2048
EFFNET_DIM    = 1280
F3NET_DIM     = 1024
VIT_DIM       = 768
SWIN_DIM      = 768       
CAPSNET_DIM   = 256
UNET_DIM      = 512
FORENSIC_DIM  = 1296      

PROJ_DIM      = 256
ATTN_DIM      = 128        

STREAM_DIMS = [
    XCEPTION_DIM,
    EFFNET_DIM,
    F3NET_DIM,
    VIT_DIM,
    SWIN_DIM,
    CAPSNET_DIM,
    UNET_DIM,
    FORENSIC_DIM,
]
