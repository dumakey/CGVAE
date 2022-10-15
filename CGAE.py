# -*- coding: utf-8 -*-

import os
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from collections import OrderedDict

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()

import DL_models as models
from Preprocessing import ImageTransformer
import AugmentationDataset as ADS
import reader


class CGenTrainer:

    def __init__(self, launch_file):

        class parameter_container:
            pass
        class dataset_container:
            pass
        class model_container:
            pass
        class predictions_container:
            pass

        self.parameters = parameter_container()
        self.datasets = dataset_container()
        self.model = model_container()
        self.predictions = predictions_container()

        # Setup general parameters
        casedata = reader.read_case_setup(launch_file)
        self.parameters.analysis = casedata.analysis
        self.parameters.training_parameters = casedata.training_parameters
        self.parameters.img_processing = casedata.img_processing
        self.parameters.img_size = casedata.img_resize
        self.parameters.samples_generation = casedata.samples_generation
        self.parameters.data_augmentation = casedata.data_augmentation
        self.case_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [parameter for parameter in self.parameters.training_parameters.items() if type(parameter[1]) == list]
        if len(sens_vars) != 0:
            self.parameters.sens_variable = sens_vars[0]
        else:
            self.parameters.sens_variable = None

        # Check for model reconstruction
        if self.parameters.analysis['import'] == True:
            self.model.imported = True
            self.reconstruct_model()
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to generate contours based on Bayesian Deep learning algorithms'.format(class_name)

    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'SINGLETRAINING': self.singletraining,
                        'SENSANALYSIS': self.sensitivity_analysis_on_training,
                        'GENERATE': self.contour_generation,
                        'DATAGEN': self.data_generation,
                        }

        analysis_list[analysis_ID]()

    def sensitivity_analysis_on_training(self):

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        # Perform sensitivity analysis
        self.set_datasets()
        self.set_tensorflow_datasets()
        self.train_model(sens_variable)
        self.export_nn_log()
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)

    def singletraining(self):

        self.set_datasets()
        self.set_tensorflow_datasets()
        self.train_model()
        self.export_nn_log()
        self.export_model_performance()
        self.export_model()

    def data_generation(self):

        transformations = [{k:v[1:] for (k,v) in self.parameters.img_processing.items() if v[0] == 1}][0]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)

    def contour_generation(self):

        if self.model.imported == False:
            self.singletraining()

        ## GENERATE NEW DATA - SAMPLING ##
        X_samples = self.generate_samples()
        print()

        # Show the sampled images
        n_samples = X_samples.shape[0]
        width, height = self.parameters.img_size
        plt.figure()
        for i in range(n_samples):
            ax = plt.subplot(n_samples//5 + 1,5,i+1)
            plt.imshow(X_samples[i,:].reshape((width,height)), cmap='gray')
            ax.axis('off')
        #plt.show()
        plt.savefig(os.path.join(self.case_dir,'Results','generated_samples.png'), dpi=100)
        plt.close()
        print()

    def preprocess_data(self, im_tilde, im):

        im_tilde_tf = tf.cast(im_tilde,tf.float32)
        im_tilde_tf = im_tilde_tf/255

        im_tf = tf.cast(im,tf.float32)
        im_tf = im_tf/255

        return im_tilde_tf, im_tf

    def read_dataset(self, dataset_folder='Training', format='png'):

        img_filepaths = []
        for (root, case_dirs, _) in os.walk(os.path.join(self.case_dir,'Datasets',dataset_folder)):
            for case_dir in case_dirs:
                files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir)) if file.endswith(format)]
                img_filepaths += files

        img_list = []
        for filepath in img_filepaths:
            img = cv.imread(filepath)
            gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            img_list.append(gray_img)

        return img_list

    def preprocess_image(self, img):

        img_dimensions = self.parameters.img_size
        m = len(img)
        imgs_processed = np.zeros((m,img_dimensions[1]*img_dimensions[0]),dtype=np.float32)
        #imgs_processed = np.zeros((m,img_dimensions[1],img_dimensions[0]),dtype=np.float32)
        for i in range(m):
            if img[i].shape[0:2] != (img_dimensions[1],img_dimensions[0]):
                img_processed = ImageTransformer.resize(img[i],img_dimensions)
            else:
                img_processed = img[i]
            img_processed = cv.bitwise_not(img_processed)
            imgs_processed[i] = img_processed.reshape((np.prod(img_processed.shape[0:])))
            #imgs_processed[i] = img_processed

        return imgs_processed

    def set_datasets(self):

        # Read original datasets
        X = self.read_dataset()
        # Resize images, if necessary
        X = self.preprocess_image(X)

        X_train, X_val = train_test_split(X,train_size=self.parameters.training_parameters['train_size'],shuffle=True)
        X_cv, X_test = train_test_split(X_val,train_size=0.75,shuffle=True)

        self.datasets.data_train = (X_train, X_train)
        self.datasets.data_cv = (X_cv, X_cv)
        self.datasets.data_test = (X_test, X_test)

    def create_dataset_pipeline(self, dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):

        dataset_tensor = tf.data.Dataset.from_tensor_slices(dataset)

        if is_train:
            dataset_tensor = dataset_tensor.shuffle(buffer_size=dataset[0].shape[0]).repeat()
        dataset_tensor = dataset_tensor.map(self.preprocess_data,num_parallel_calls=num_threads)
        dataset_tensor = dataset_tensor.batch(batch_size)
        dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

        return dataset_tensor

    def set_tensorflow_datasets(self):

        batch_size_train = self.parameters.training_parameters['batch_size']
        self.datasets.dataset_train = self.create_dataset_pipeline(self.datasets.data_train,is_train=True,
                                                                   batch_size=batch_size_train)
        self.datasets.dataset_cv = self.create_dataset_pipeline(self.datasets.data_cv,is_train=False,batch_size=1)
        self.datasets.dataset_test = self.preprocess_data(self.datasets.data_test[0],self.datasets.data_test[1])

    def generate_augmented_data(self, transformations, augmented_dataset_size=1):

        # Set storage folder for augmented dataset
        augmented_dataset_dir = os.path.join(self.case_dir,'Datasets','Augmented')

        img_dims = self.parameters.img_size
        # Unpack data
        X = self.read_dataset(dataset_folder='To_augment')
        # Generate new dataset
        data_augmenter = ADS.datasetAugmentationClass(X,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def train_model(self, sens_var=None):

        # Parameters
        input_dim = self.parameters.img_size
        latent_dim = self.parameters.training_parameters['latent_dim']
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        batch_size = self.parameters.training_parameters['batch_size']
        l2_reg = self.parameters.training_parameters['l2_reg']
        l1_reg = self.parameters.training_parameters['l1_reg']
        dropout = self.parameters.training_parameters['dropout']

        if self.model.imported == False:
            pretrained_model = None
        else:
            pretrained_model = self.model.Model

        if sens_var == None:  # If it is a one-time training
            self.model.Model = models.VAE(input_dim,latent_dim,alpha,l2_reg,l1_reg,dropout,mode='train',model='mixed')
            self.model.History = self.model.Model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                      validation_data=self.datasets.dataset_cv,validation_steps=None)
            '''
            ## TRAIN ##
            n_samples = 10
            width, height = self.parameters.img_size
            ## PLOT GENERATED TRAINING DATA ##
            fig, ax = plt.subplots(5,2,sharex=True,figsize=(10,10))
            for fid_idx, (data, title) in enumerate(
                    zip([self.datasets.data_train[0],self.datasets.data_test[0]], ['Train','Test'])):
                ii = 0
                for i in range(n_samples//2):
                    for j in range(2):
                        x_dec = self.model.Model.predict(data[ii,:].reshape((1,height*width)))
                        x_dec = np.reshape(x_dec,(width,height))
                        ax[i,j].imshow(x_dec,cmap='Greys_r')
                        fig_set = fig.add_subplot(5,2,fid_idx + 1)
                        fig_set.set_title(title)
                        fig_set.axis('off')
                        ii += 1
            #plt.show()
            plt.savefig(os.path.join(self.case_dir,'Results','training_samples.png'),dpi=100)
            '''

        else: # If it is a sensitivity analysis
            self.model.Model = []
            self.model.History = []
            if type(alpha) == list:
                for learning_rate in alpha:
                    if self.model.imported == False:
                        model = models.VAE(input_dim,latent_dim,learning_rate,l2_reg,l1_reg,dropout,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                              validation_data=self.datasets.dataset_cv,validation_steps=None))

            elif type(l2_reg) == list:
                for regularizer in l2_reg:
                    if self.model.imported == False:
                        model = models.VAE(input_dim,latent_dim,alpha,regularizer,l1_reg,dropout,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None))
            elif type(l1_reg) == list:
                for regularizer in l1_reg:
                    if self.model.imported == False:
                        model = models.VAE(input_dim,latent_dim,alpha,l2_reg,regularizer,dropout,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None))
            elif type(dropout) == list:
                for rate in dropout:
                    if self.model.imported == False:
                        model = models.VAE(input_dim,latent_dim,alpha,l2_reg,l1_reg,rate,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None))


    def generate_samples(self):

        ## BUILD DECODER ##
        n_samples = self.parameters.samples_generation['n_samples']
        output_dim = self.model.Model.input.shape[1]
        latent_dim = self.parameters.training_parameters['latent_dim']
        alpha = self.parameters.training_parameters['learning_rate']
        decoder = models.VAE(output_dim,latent_dim,alpha,dropout=0.0,l2_reg=0.0,l1_reg=0.0,mode='sample')  # No regularization
        # Retrieve decoder weights
        model_weights = self.model.Model.weights
        j = 0
        for weight in model_weights:
            j_layer_shape = weight.get_shape()[0]
            if j_layer_shape != latent_dim:
                j += 1
            else:
                break
        decoder_input_layer_idx = j

        decoder_weights = self.model.Model.get_weights()[decoder_input_layer_idx:]
        decoder.set_weights(decoder_weights)

        ## SAMPLE IMAGES ##
        t = tf.random.normal(shape=(1,latent_dim))
        X_samples = np.zeros([n_samples,output_dim])
        for i in range(n_samples):
            X_samples[i,:] = decoder.predict(t,steps=1)

        return X_samples

    def export_model_performance(self, sens_var=None):

        try:
            History = self.model.History
        except:
            raise Exception('There is no evolution data for this model. Train model first.')
        else:
            if type(History) == list:
                N = len(History)
            else:
                N = 1
                History = [History]

            # Loss evolution plots #
            Nepochs = self.parameters.training_parameters['epochs']
            epochs = np.arange(1,Nepochs+1,1)

            case_ID = self.parameters.analysis['case_ID']
            for i,h in enumerate(History):
                loss_train = h.history['loss']
                loss_cv = h.history['val_loss']

                fig, ax = plt.subplots(1)
                ax.plot(epochs,loss_train,label='Training',color='r')
                ax.plot(epochs,loss_cv,label='Cross-validation',color='b')
                ax.grid()
                ax.set_xlabel('Epochs',size=12)
                ax.set_ylabel('Loss',size=12)
                ax.tick_params('both',labelsize=10)
                ax.legend()

                if sens_var:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance',
                                               '{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance')
                if os.path.exists(storage_dir):
                    rmtree(storage_dir)
                os.makedirs(storage_dir)
                fig.savefig(os.path.join(storage_dir,'Loss_evolution.png'),dpi=200)
                plt.close()

                # Metrics #
                metrics_name = [item for item in h.history if item not in ('loss','val_loss')]
                metrics_val = [(metric,h.history[metric][0]) for metric in metrics_name if metric.startswith('val')]
                metrics_train = [(metric,h.history[metric][0]) for metric in metrics_name if not metric.startswith('val')]

                rows = [metric[0] for metric in metrics_train]
                metric_fun = lambda L: np.array([item[1] for item in L])
                metrics_data = np.vstack((metric_fun(metrics_train),metric_fun(metrics_val))).T
                metrics = pd.DataFrame(index=rows,columns=['Training','CV'],data=metrics_data)
                metrics.to_csv(os.path.join(storage_dir,'Model_metrics.csv'),sep=';',decimal='.')

                # Loss
                loss_data = np.vstack((list(epochs), loss_train, loss_cv)).T
                loss = pd.DataFrame(columns=['Epoch', 'Training', 'CV'], data=loss_data)
                loss.to_csv(os.path.join(storage_dir, 'Model_loss.csv'), index=False, sep=';', decimal='.')

    def export_model(self, sens_var=None):

        if type(self.model.Model) == list:
            N = len(sens_var[1])
            model = self.model.Model
        else:
            N = 1
            model = [self.model.Model]

        case_ID = self.parameters.analysis['case_ID']
        for i in range(N):
            if sens_var:
                weights_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
            else:
                weights_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_{}')
            if os.path.exists(weights_dir):
                rmtree(weights_dir)
            os.makedirs(weights_dir)

            # Export model arquitecture to JSON file
            model_json = model[i].to_json()
            with open(os.path.join(weights_dir,'CGAE_model_arquitecture.json'),'w') as json_file:
                json_file.write(model_json)

            # Export model weights to HDF5 file
            model[i].save_weights(os.path.join(weights_dir,'CGAE_model_weights.h5'))

    def reconstruct_model(self):

        weights_dir = os.path.join(self.case_dir,'Results','pretrained_Model')
        # Load JSON file
        json_file = open(os.path.join(weights_dir,'CGAE_model_arquitecture.json'),'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # Build model
        self.model.Model = tf.keras.models.model_from_json(loaded_model_json)
        # Load weights into new model
        self.model.Model.load_weights(os.path.join(weights_dir,'CGAE_model_weights.h5'))

    def export_nn_log(self):

        training = OrderedDict()
        training['TRAINING SIZE'] = self.parameters.training_parameters['train_size']
        training['LEARNING RATE'] = self.parameters.training_parameters['learning_rate']
        training['L2 REGULARIZER'] = self.parameters.training_parameters['l2_reg']
        training['L1 REGULARIZER'] = self.parameters.training_parameters['l1_reg']
        training['DROPOUT'] = self.parameters.training_parameters['dropout']
        training['NUMBER OF EPOCHS'] = self.parameters.training_parameters['epochs']
        training['BATCH SIZE'] = self.parameters.training_parameters['batch_size']
        training['LATENT DIMENSION'] = self.parameters.training_parameters['latent_dim']
        training['OPTIMIZER'] = self.model.Model.optimizer._name
        training['METRICS'] = [self.model.Model.metrics_names[0] if self.model.Model.metrics_names != None else None]
    
        analysis = OrderedDict()
        analysis['CASE ID'] = self.parameters.analysis['case_ID']
        analysis['ANALYSIS'] = self.parameters.analysis['type']
        analysis['IMPORTED MODEL'] = self.parameters.analysis['import']
        analysis['LAST TRAINING LOSS'] = self.model.History.history['loss'][-1]
        analysis['LAST CV LOSS'] = self.model.History.history['val_loss'][-1]

        architecture = OrderedDict()
        architecture['NUMBER OF INPUTS'] = len(self.model.Model.inputs),
        architecture['INPUT SHAPE'] = self.datasets.dataset_train.element_spec[0].shape,
        architecture['LAYERS'] = [layer._name for layer in self.model.Model.layers],
        architecture['TRAINABLE VARIABLES'] = sum([np.prod(variable.shape) for variable in self.model.Model.trainable_variables]),
        architecture['NON TRAINABLE VARIABLES'] = sum([np.prod(variable.shape) for variable in self.model.Model.non_trainable_variables]),

        case_ID = self.parameters.analysis['case_ID']
        storage_folder = os.path.join(self.case_dir,'Results',str(case_ID))
        with open(os.path.join(storage_folder,'CGAE.log'),'w') as f:
            f.write('CGAE log file\n')
            f.write('==========\n')
            f.write('->ANALYSIS\n')
            for item in analysis.items():
                f.write('>' + item[0] + '=' + str(item[1]) + '\n')
            f.write('-----\n')
            f.write('->TRAINING\n')
            for item in training.items():
                f.write('>' + item[0] + '=' + str(item[1]) + '\n')
            f.write('==========\n')

if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\Contour_generator\Scripts\launcher.dat'
    trainer = CGenTrainer(launcher)
    trainer.launch_analysis()
    print()