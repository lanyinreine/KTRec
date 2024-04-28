def get_exp_configure(model):
    configure_list = {
        'input_size': 128,
        'hidden_size': 128,
        'output_size': 1,
        'batch_size': 128,
        'dropout': 0.5,
        'decay_step': 100,
        'l2_reg': 1e-4,
        'pre_hidden_sizes': [256, 64, 16],
        'retrieval': False,
        'forRec': False
    }
    if model == 'CoKT':
        configure_list.update({'batch_size': 16, 'retrieval': True})
    if model == 'GRU4Rec':
        configure_list.update({'forRec': True})
    return configure_list
