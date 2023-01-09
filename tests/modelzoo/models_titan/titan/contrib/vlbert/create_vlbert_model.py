
import sys
from pathlib import Path
from titan.utils.misc import download_and_extract_scm_permission

exist_vlbert = any('vlbert' in path for path in sys.path)
if not exist_vlbert:
    # requires vlbert: https://code.byted.org/shicheng.cecil/vlbert
    vlbert_dest = '/opt/tiger/vlbert'
    if not Path(vlbert_dest).exists():
        # Download SCM
        download_and_extract_scm_permission(vlbert_dest, 'lab.moderation.vlbert', '1.0.0.183')
    sys.path.insert(0, vlbert_dest)

sys.path.append('/opt/tiger/vlbert/')
from haggs.core import init_model, init_resume


def create_vlbert_model(config):
        # checkpoint = init_resume(config) # TODO: will support
        model = init_model(config, resume_checkpoint=None)

        return model