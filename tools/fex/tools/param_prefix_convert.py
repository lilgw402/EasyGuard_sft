
from fex.utils.torch_io import load as torch_io_load
from fex.utils.torch_io import save as torch_io_save


def convert(from_path, to_path, prefix_change):
    pretrain_state_dict = torch_io_load(
        from_path, map_location=lambda storage, loc: storage)
    pretrain_state_dict = pretrain_state_dict['state_dict'] if 'state_dict' in pretrain_state_dict else pretrain_state_dict
    pretrain_state_dict_parsed = {}
    prefix_change = [prefix_change.split('->') for prefix_change in prefix_change]

    for k, v in pretrain_state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        no_match = True
        for pretrain_prefix, new_prefix in prefix_change:
            if k.startswith(pretrain_prefix):
                k = new_prefix + k[len(pretrain_prefix):]
                pretrain_state_dict_parsed[k] = v
                no_match = False
                break
        if no_match:
            pretrain_state_dict_parsed[k] = v
    torch_io_save(pretrain_state_dict_parsed, to_path)


def convert_clip():
    f = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/abrt_224_dycover_20210915/model_old.th'
    t = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/abrt_224_dycover_20210915/model.th'
    # pm = [
    #     'albert->text',
    #     'projector->t_projector',
    #     'resnet->visual',
    #     'fc128->v_projector',
    #     'calc_nce_loss.tau->calc_nce_loss.tau'
    # ]
    pm = [
        'albert->text',
        'projector->t_projector',
        'resnet->visual',
        'fc512->v_projector.0',
        'fc128->v_projector.2',
        'calc_nce_loss.tau->calc_nce_loss.tau'
    ]
    convert(f, t, pm)


if __name__ == '__main__':
    convert_clip()
