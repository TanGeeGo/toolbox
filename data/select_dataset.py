'''
# --------------------------------------------
# select dataset
# --------------------------------------------
'''

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D # Low-quality image and High-quality image
    elif dataset_type in ['multimodal']:
        from data.dataset_multimodal import DatasetMultiModal as D
    elif dataset_type in ['multiin']:
        from data.dataset_multiin import DatasetMultiin as D

    dataset = D(dataset_opt)
    
    return dataset
