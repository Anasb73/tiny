import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import tensorflow as tf
import numpy as np
import PIL.Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Function read_train_dataset
### Returns train and valid dataset information:
###   x: list of filenames containing input data
###   y: list of expected output (ground truth)
###   The same indexes of x and y correspond to the same set
def read_train_dataset(nbOutput, datasetPath, validationSplit, maxDsetInputs=None):
    try:
        fname = datasetPath + "/annotation/train_annotation.csv"
        f_in = open(fname, "r")
        annotations = f_in.read().splitlines()
        print("*** Info: Found %d training annotations" % len(annotations))
    except IOError:
        print("*** Error: unable to open " + fname + "!")
        exit(1)

    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    data_cnt = 0

    random.shuffle(annotations) # Randomize the list of dataset inputs
    
    for line in annotations:
        if maxDsetInputs > 0 and data_cnt >= maxDsetInputs:
            break

        line_split = line.strip().split(',')
        if np.random.random() < validationSplit:
            valid_x.append(datasetPath + "/train/" + line_split[0])
            valid_y.append([float(x) for x in line_split[1:]])
        else:
            train_x.append(datasetPath + "/train/" + line_split[0])
            train_y.append([float(x) for x in line_split[1:]])
        data_cnt += 1
            
    if nbOutput != None and np.shape(valid_y)[1] != nbOutput:
        print("ERROR: specified number of output data different from dataset!")
        exit(1)
        
    return train_x, train_y, valid_x, valid_y


### Function read_test_dataset
### Returns n_data test dataset information (if n_data is None, returns full dataset).
###   x: list of filenames containing input data
###   y: list of expected output (ground truth)
###   The same indexes of x and y correspond to the same set
def read_test_dataset(datasetPath, n_data=None):
    try:
        fname = datasetPath + "/annotation/test_annotation.csv"
        f_in = open(fname, "r")
        annotations = f_in.read().splitlines()
        print("*** Info: Found %d testing annotations" % len(annotations))
    except IOError:
        print("*** Error: unable to open " + fname + "!")
        exit(1)

    test_x = []
    test_y = []
    # Do not randomize the list of dataset inputs for testing
        
    for line in annotations:
        line_split = line.strip().split(',')
        test_x.append(datasetPath + "/test/" + line_split[0])
        test_y.append([float(x) for x in line_split[1:]])
        
    return test_x, test_y


### Function write_tf_reference
### Run model on test_x input data and write the reference file
def write_tf_reference(model, test_x, nb_dump, dset_itype, rand_seed, fname):
    try:
        f_res = open(fname, "w")
    except IOError:
        print("*** ERROR: unable to open " + fname + " in write mode!")
        exit(1)

    f_res.write("%s\n" % dset_itype)
    f_res.write("%d\n" % nb_dump)
    f_res.write("%d\n" % rand_seed)
    print("Running %d inferences" % nb_dump)
    predictions = model.predict(test_x[0:nb_dump])
    print("Writing %s" % fname)
    for i in range(predictions.shape[0]):
        flat_predictions = predictions[i].flatten()
        line = ' '.join(map(str, flat_predictions))
        f_res.write(line + '\n')
    f_res.close()

    
### Function apply_softmax
### Apply softmax to input data
def apply_softmax(data):
    accum = 0
    for d in data:
        accum += np.exp(d)
    for i in range(len(data)):
        data[i] = np.exp(data[i]) / accum

        
### Function apply_sigmoid
### Apply sigmoid  to all input data
def apply_sigmoid(data):
    for i in range(len(data)):
        data[i] = 1 / (1 + np.exp(-data[i]))

        
### Class DataGenWrapper
### Ease input data handling (batches, data augmentation ...)
### x_type: "image" or "ascii"
class DataGenWrapper(tf.keras.utils.Sequence):
    def __init__(self, list_x_paths, list_y, x_type="image", input_shape=None, output_shape=None,
                 input_scaling=1.0, batch_size=32, standardize=True, augmentation=True, shuffle=True):
        self.list_x_paths = list_x_paths
        self.list_y = list_y
        self.x_type = x_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_scaling = input_scaling
        self.batch_size = batch_size
        self.standardize = standardize
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.on_epoch_end()
        self.datagen = ImageDataGenerator(
          featurewise_center=False,
          featurewise_std_normalization=False,
          rotation_range=20,
          #width_shift_range=0.3,
          #height_shift_range=0.3,
          horizontal_flip=True)

    def __len__(self):
        return len(self.list_x_paths) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__get_data(batch_indices)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_x_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def standardize_image_float(self, x):
        x -= np.mean(x, keepdims=True)
        x /= ((np.std(x, keepdims=True) + 1e-3))
        return x

    def standardize_image_int(self, x, nbit_data=8):
        x -= np.round(np.mean(x, keepdims=True))
        x += 2**(nbit_data-1)
        x = np.clip(x, 0, 2**nbit_data-1).astype('int')
        return x

    def show_image_with_bbox(self, image, bbox=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if bbox is not None:
            ax.add_patch(
                 patches.Rectangle(
                    (bbox[0],bbox[1]),
                    bbox[2],
                    bbox[3],
                    edgecolor = 'red',
                    facecolor = 'none',
                    fill=True
                    ) )
        #ax.axis("off")
        ax.imshow(image, cmap='gray')
        plt.show()

        
    def get_samples(self, n=None, shuffle=False):
        all_x = None
        all_y = None

        if n is None:
            n_samples = len(self.list_x_paths)
            n_iter = n_samples // self.batch_size
        else:
            n_samples = n
            n_iter = (n_samples + self.batch_size - 1) // self.batch_size

        indices = np.arange(n_iter * self.batch_size)
        if shuffle:
            np.random.shuffle(indices)
            
        for i in range(n_iter):
            x, y = self.__get_data(indices[i*self.batch_size: (i+1)*self.batch_size])
            if all_x is None:
                all_x = x
                all_y = y
            else:
                all_x = np.vstack((all_x, x))
                all_y = np.vstack((all_y, y))
                
        if n is None:
            return all_x, all_y
        else:
            return all_x[:n_samples], all_y[:n_samples]
        

    def __get_data(self, batch_indices):
        if self.x_type == "random":
            x_batch = np.random.rand(self.batch_size, self.input_shape[0],
                                     self.input_shape[1], self.input_shape[2])
            if isinstance(self.output_shape, int):
                y_batch = np.random.rand(self.batch_size, self.output_shape)
            elif len(self.output_shape) == 1:
                y_batch = np.random.rand(self.batch_size, self.output_shape[0])
            else:
                y_batch = np.random.rand(self.batch_size, self.output_shape[0],
                                         self.output_shape[1], self.output_shape[2])
            return x_batch, y_batch
            
        for i, id in enumerate(batch_indices):
            if self.x_type == "image":
                inputs_array = np.array(PIL.Image.open(self.list_x_paths[id]))
                x = inputs_array.astype('float')
                x = x[..., None] if len(x.shape) == 2 else x # Add channel dimension
                ## In case of image and in case an input shape is specified, reshape it
                if self.input_shape is not None:
                    x_new = np.zeros(self.input_shape)
                    min_shape = [min(x.shape[i],self.input_shape[i]) for i in range(3)]
                    x_new[:min_shape[0], :min_shape[1], :min_shape[2]] = x[:min_shape[0], :min_shape[1], :min_shape[2]]
                    x = x_new
                    
            elif self.x_type == "ascii":
                f_in = open(self.list_x_paths[id])
                all_lines = f_in.readlines()
                f_in.close()
                x = []
                for line in all_lines:
                    x_line = [float(x) for x in line.strip().split(',')]
                    x_line += [0] * (self.input_shape[1] - len(x_line))
                    x.append(x_line)
                x = np.array(x)
                x = x[..., None] if len(x.shape) == 2 else x # Add channel dimension
                
            else:
                print("ERROR: x_type not supported (%s)" % self.x_type)
                exit(1)

            if self.output_shape is None or isinstance(self.output_shape, int) or len(self.output_shape) == 1:
                y = self.list_y[id]
            elif len(self.output_shape) == 3:
                y = np.reshape(self.list_y[id], self.output_shape)
            else:
                print("ERROR: output shape must have 1 or 3 dimensions")
                exit(1)
                      
            if i == 0:
                if self.output_shape is None or isinstance(self.output_shape, int) or len(self.output_shape) == 1:
                    y_batch = np.empty((self.batch_size, len(y)))
                else:
                    y_batch = np.empty((self.batch_size, y.shape[0], y.shape[1], y.shape[2]))
                    
                if len(x.shape) == 2:
                    x_batch = np.empty((self.batch_size, x.shape[0], x.shape[1], 1))
                else:
                    x_batch = np.empty((self.batch_size, x.shape[0], x.shape[1], x.shape[2]))
                    
            ### Center, normalize and scale input data
            if self.standardize:
                x = self.standardize_image_int(x, 8)
            x = x / self.input_scaling
                
            ### Data augmentation: blur
            #if self.augmentation:
            #    x = gaussian_filter(x, np.random.rand()*1.0)
                
            ### Data augmentation: TF
            if self.augmentation:
              x = self.datagen.random_transform(x)
              
            #self.show_image_with_bbox(x, np.array(y[1:]))

            ### Append to current batch
            x_batch[i] = x
            y_batch[i] = y
        return x_batch, y_batch
