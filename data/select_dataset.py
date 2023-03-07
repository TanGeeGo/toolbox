'''
# --------------------------------------------
# select dataset
# --------------------------------------------
'''

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset