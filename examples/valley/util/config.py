IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

MULTI_IM_START_TOKEN = [DEFAULT_IM_START_TOKEN] + [f"<im_start{id}>" for id in range(1,10)]
MULTI_IM_END_TOKEN = [DEFAULT_IM_END_TOKEN] + [f"<im_end{id}>" for id in range(1,10)]

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_FRAME_TOKEN = "<vi_frame>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"
DEFAULT_GANDALF_TOKEN = "<gandalf>"

DEFAULT_EOC_TOKEN = "<eoc>"