import re

def read_case_setup(launch_filepath):
    file = open(launch_filepath, 'r')
    data = file.read()

    class setup:
        pass

    casedata = setup()
    casedata.case_dir = None
    casedata.analysis = dict.fromkeys(['type', 'import'], None)
    casedata.training_parameters = \
        dict.fromkeys(['train_size', 'learning_rate', 'l2_reg', 'l1_reg', 'dropout', 'epochs', 'batch_size'], None)
    casedata.img_processing = {'slice_size': [None, None],
                               }
    casedata.samples_generation = {'n_samples': None}

    ## Data directory
    match = re.search('DATADIR\s*=\s*(.*).*', data)
    if match:
        casedata.case_dir = match.group(1)

    ## Analysis
    # Type of analysis
    match = re.search('TYPEANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['type'] = match.group(1)

    # Import
    match = re.search('IMPORTMODEL\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['import'] = int(match.group(1))

    ## Training parameters
    # Latent dimension
    match = re.search('LATENTDIM\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['latent_dim'] = None
        else:
            casedata.training_parameters['latent_dim'] = int(match.group(1))

    # Training dataset size
    match = re.search('TRAINSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['train_size'] = 0.75
        else:
            casedata.training_parameters['train_size'] = float(match.group(1))

    # Learning rate
    match = re.search('LEARNINGRATE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['learning_rate'] = 0.001
        else:
            casedata.training_parameters['learning_rate'] = float(match.group(1))

    # L2 regularizer
    match = re.search('L2REG\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['l2_reg'] = 0.0
        else:
            casedata.training_parameters['l2_reg'] = float(match.group(1))

    # L1 regularizer
    match = re.search('L1REG\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['l1_reg'] = 0.0
        else:
            casedata.training_parameters['l1_reg'] = float(match.group(1))

    # Dropout
    match = re.search('DROPOUT\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['dropout'] = 0.0
        else:
            casedata.training_parameters['dropout'] = float(match.group(1))

    # Number of epochs
    match = re.search('EPOCHS\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['epochs'] = 1
        else:
            casedata.training_parameters['epochs'] = int(match.group(1))

    # Batch size
    match = re.search('BATCHSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['batch_size'] = None
        else:
            casedata.training_parameters['batch_size'] = int(match.group(1))

    ## Image processing parameters
    # Image resize
    match_dist = re.search('IMAGERESIZE\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
    if match_dist:
        casedata.img_processing['slice_size'][0] = int(match_dist.group(1))
        casedata.img_processing['slice_size'][1] = int(match_dist.group(2))
        casedata.img_processing['slice_size'] = tuple(casedata.img_processing['slice_size'])

    ## Sample generation parameters
    # Number of samples
    match = re.search('NSAMPLES\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.samples_generation['n_samples'] = 1
        else:
            casedata.samples_generation['n_samples'] = int(match.group(1))

    return casedata