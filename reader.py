import re
from random import randint

def read_case_setup(launch_filepath):
    file = open(launch_filepath, 'r')
    data = file.read()

    class setup:
        pass

    casedata = setup()
    casedata.case_dir = None
    casedata.analysis = dict.fromkeys(['case_ID','type', 'import'], None)
    casedata.training_parameters = dict()
    casedata.img_resize = [None,None]
    casedata.img_processing = {'rotation': [None, None, None, None],
                               'translation': [None, None, None],
                               'zoom': [None, None],
                               'filter': [None, None, None, None, None],
                               'flip': [None, None]
                               }
    casedata.samples_generation = {'n_samples': None}
    casedata.activation_plotting = {'n_samples': None, 'n_cols': None, 'rows2cols_ratio': None}
    casedata.data_augmentation = [None, None]

    ############################################### Data directory #####################################################
    match = re.search('DATADIR\s*=\s*(.*).*', data)
    if match:
        casedata.case_dir = match.group(1)

    ################################################## Analysis ########################################################
    casedata.analysis['case_ID'] = randint(1,9999)
    # Type of analysis
    match = re.search('TYPEANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['type'] = match.group(1)

    # Import
    match = re.search('IMPORTMODEL\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['import'] = int(match.group(1))

    ## Dataset augmentation
    match = re.search('AUGDATA\s*=\s*(\d).*', data)
    if match:
        casedata.data_augmentation[0] = int(match.group(1))
        match_factor = re.search('AUGDATASIZE\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.data_augmentation[1] = float(match_factor.group(1))

    ############################################# Training parameters ##################################################
    # Latent dimension
    match = re.search('LATENTDIM\s*=\s*(.*)', data)
    if match:
        matches = re.findall('(\d+)',match.group(1))
        casedata.training_parameters['latent_dim'] = int(matches[0]) if len(matches) == 1 else [int(item) for item in matches]

    # Encoder hidden dimension
    match = re.search('ENCHIDDENDIM\s*=\s*\(\s*((\d+)\s*,?\s*(\d+)*)\s*\).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['enc_hidden_layers'] = [None]
        else:
            casedata.training_parameters['enc_hidden_layers'] = [int(item) for item in match.groups()[1:] if item != None]

    # Decoder hidden dimension
    match = re.search('DECHIDDENDIM\s*=\s*\(\s*((\d+)\s*,?\s*(\d+)*)\s*\).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['dec_hidden_layers'] = [None]
        else:
            casedata.training_parameters['dec_hidden_layers'] = [int(item) for item in match.groups()[1:] if item != None]

    # Training dataset size
    match = re.search('TRAINSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['train_size'] = 0.75
        else:
            casedata.training_parameters['train_size'] = float(match.group(1))

    # Learning rate
    match = re.search('LEARNINGRATE\s*=\s*(.*)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        casedata.training_parameters['learning_rate'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]

    # L2 regularizer
    match = re.search('L2REG\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l2_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l2_reg'] = 0.0

    # L1 regularizer
    match = re.search('L1REG\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['l1_reg'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['l1_reg'] = 0.0

    # Dropout
    match = re.search('DROPOUT\s*=\s*(.*|NONE)', data)
    if match:
        matches = re.findall('(\d+\.?\d*)',match.group(1))
        if matches:
            casedata.training_parameters['dropout'] = float(matches[0]) if len(matches) == 1 else [float(item) for item in matches]
        else:
            casedata.training_parameters['dropout'] = 0.0

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

    # Activation function
    match = re.search('ACTIVATION\s*=\s*((\w+)\s*,?\s*(\w+)*)\s*.*', data)
    if match:
        if None in match.groups():
            casedata.training_parameters['activation'] = str.lower(match.group(2))
        else:
            casedata.training_parameters['activation'] = [str.lower(item) for item in match.groups()[1:]]

    # NN architecture
    match = re.search('ARCHITECTURE\s*=\s*(\w+).*', data)
    if match:
        casedata.training_parameters['architecture'] = str.lower(match.group(1))

    ######################################## Image processing parameters ###############################################
    # Image resize
    match_dist = re.search('IMAGERESIZE\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
    if match_dist:
        casedata.img_resize[0] = int(match_dist.group(1))
        casedata.img_resize[1] = int(match_dist.group(2))
        casedata.img_resize = tuple(casedata.img_resize)

    # Rotation
    match = re.search('ROTATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['rotation'][0] = int(match.group(1))
        match_angle = re.search('ROTATIONANGLE\s*=\s*([\+|\-]?\d+\.?\d*).*', data)
        if match_angle:
            casedata.img_processing['rotation'][1] = float(match_angle.group(1))
        match = re.search('ROTATIONCENTER\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
        if match:
            if match.group(1) != 'NONE':
                casedata.img_processing['rotation'][2] = int(match.group(1))
            elif match.group(2) != 'NONE':
                casedata.img_processing['rotation'][3] = int(match.group(2))

    # Translation
    match = re.search('TRANSLATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['translation'][0] = int(match.group(1))
        match_dist = re.search('TRANSLATIONDIST\s*=\s*\(([\+|\-]?\d+|NONE)\,+([\+|\-]?\d+|NONE)\).*', data)
        if match_dist:
            casedata.img_processing['translation'][1] = float(match_dist.group(1))
            casedata.img_processing['translation'][2] = float(match_dist.group(2))

    # Zoom
    match = re.search('ZOOM\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['zoom'][0] = int(match.group(1))
        match_factor = re.search('ZOOMFACTOR\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.img_processing['zoom'][1] = float(match_factor.group(1))
    # Filter
    match = re.search('FILTER\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['filter'][0] = int(match.group(1))
        match_type = re.search('FILTERTYPE\s*=\s*(\w+).*', data)
        casedata.img_processing['filter'][1] = match_type.group(1)
        if match_type:
            if match_type.group(1) == 'GAUSSIAN':
                filter_param = re.search(
                    'FILTERPARAM\s*=\s*\(\s*SIZE\s*\,\s*(\d+|NONE)\s*\,\s*SIGMA\s*\,\s*(\d+|NONE)\s*\).*', data)
                casedata.img_processing['filter'][2] = int(filter_param.group(1))
                casedata.img_processing['filter'][3] = int(filter_param.group(2))
        elif match_type.group(1) == 'BILATERAL':
            filter_param = re.search(
                'FILTERPARAM\s*=\s*\(\s*(D)\s*\,\s*(\d+|NONE)\s*\,\s*SIGMACOLOR\s*\,\s*(\d+|NONE)\s*SIGMASPACE\s*\,\s*(\d+|NONE)\s*\).*',
                data)
            casedata.img_processing['filter'][2] = int(filter_param.group(1))
            casedata.img_processing['filter'][3] = int(filter_param.group(2))
            casedata.img_processing['filter'][4] = int(filter_param.group(3))

        # Flip
        match = re.search('FLIP\s*=\s*(\d).*', data)
        if match:
            casedata.img_processing['flip'][0] = int(match.group(1))
            match_type = re.search('FLIPTYPE\s*=\s*(\w+).*', data)
            if match_type:
                casedata.img_processing['flip'][1] = match_type.group(1)

    ######################################### Sample generation parameters #############################################
    # Number of samples
    match = re.search('NSAMPLESGEN\s*=\s*(\d+|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.samples_generation['n_samples'] = 1
        else:
            casedata.samples_generation['n_samples'] = int(match.group(1))

    ######################################### Activation plotting parameters ###########################################
    # Number of samples
    match = re.search('NSAMPLESACT\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['n_samples'] = int(match.group(1))

    # Number of columns
    match = re.search('NCOLS\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['n_cols'] = int(match.group(1))

    # Rows-to-columns figure ratio
    match = re.search('ROWS2COLS\s*=\s*(\d+).*', data)
    if match:
        casedata.activation_plotting['rows2cols_ratio'] = int(match.group(1))

    return casedata