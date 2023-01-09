import titan


def test_last_out_channels():
    for model in ['resnet50', 'swin_base', 'swin_base_moe']:
        m = titan.create_model(model)
        assert getattr(m, 'last_out_channels', 0) > 0


if __name__ == '__main__':
    test_last_out_channels()
