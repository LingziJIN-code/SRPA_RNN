 

#%%Functions for packages
import subprocess

def check_and_install_packages(packages):
    """
    Check if the given packages are installed in the current environment.
    If a package is not installed, it will be installed automatically.
    
    Args:
        packages (list): List of package names.
    """
    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.call(['conda', 'install', package, '-y'])
            print(f"{package} has been installed.")

def import_packages_with_alias(package_dict):
    """
    Import the given list of packages with specified aliases after ensuring they are installed.
    
    Args:
        package_dict (dict): Dictionary with package names as keys and their aliases as values.
    """
    packages = list(package_dict.keys())
    check_and_install_packages(packages)
    for package, alias in package_dict.items():
        globals()[alias] = __import__(package)
        
        


 
#%% Functions to calculate function values
 


def CalculY(x_dataset, A, W, V, b, c, T, Nh, Ny):
    # This function is used to calculate y^hat by A, W, V, b, c recurrently nor auxi1liary variables h and u
    for t in range(T):
        x_t = x_dataset[t, :].reshape(Nx, 1)
        if t == 0:
            ut_t = V @ x_t + b
            ut = ut_t
            ht_t = np.maximum(ut_t, 0)
            ht = ht_t
            y_hat_t = A @ ht_t + c
            y_hat = y_hat_t.T
        else:
            ut_t = W @ ht_t + V @ x_t + b
            ut = np.concatenate((ut, ut_t), axis=1)
            ht_t = np.maximum(ut_t, 0)
            ht = np.concatenate((ht, ht_t), axis=1)
            y_hat_t = A @ ht_t + c
            y_hat = np.concatenate((y_hat, y_hat_t.T), axis=0)
    u = ut.T.ravel()
    h = ht.T.ravel()
    u = u.reshape(T * Nh, 1)
    h = h.reshape(T * Nh, 1)
    return u, ut, h, ht, y_hat

 
 

 

 
 

 

 
 

def ObjValue(x_dataset, y_dataset, A, W, V, b, c, lambda1, lambda2, lambda3, lambda4, lambda5, Nx, Nh, Ny, T):
    _, _, _, _, y_hat = CalculY(x_dataset, A, W, V, b, c, T, Nh, Ny)
    y_hat_flat = y_hat.T.ravel().reshape(T * Ny, 1)
    y_dataset_flat = y_dataset.T.ravel().reshape(T * Ny, 1)
    loss = (1 / T) * np.sum((y_hat_flat - y_dataset_flat) ** 2)
    reg1 = lambda1 * np.sum(A ** 2) + lambda2 * np.sum(W ** 2) + lambda3 * np.sum(V ** 2)
    reg2 = lambda4 * np.sum(b ** 2) + lambda5 * np.sum(c ** 2)
    final_value = loss + reg1 + reg2
    return final_value





 
 
 
 
 
 

def convert_and_reshape(array, new_shape):
    tensor = tf.convert_to_tensor(array, dtype=tf.float32)
    return tf.reshape(tensor, new_shape)



        

#%% Install and import packages
## Dictionary of packages to be checked, installed if necessary, and imported with aliases
package_dict = {
    'numpy': 'np',
    'pandas': 'pd',
    'scipy': 'scipy',
    'tensorflow': 'tf',
    'keras': 'keras',
    'copy': 'copy',
    'time': 'time',
    'random': 'random',
    'math': 'math',
    'gc':'gc' 
}
import_packages_with_alias(package_dict)
# print('tf version:', tf.__version__)
#%%Callback functions for keras
class GetWeights(keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs={}):
    # this function runs at the end of each epoch
    # loop over each layer and get weights and biases
        V = np.linalg.norm(self.model.layers[0].get_weights()[0])
        W = np.linalg.norm(self.model.layers[0].get_weights()[1])
        b = np.linalg.norm(self.model.layers[0].get_weights()[2])
        A = np.linalg.norm(self.model.layers[1].get_weights()[0])
        c = np.linalg.norm(self.model.layers[1].get_weights()[1])
        # save all weights and biases inside a dictionary
        if epoch == 0:
            # create array to hold weights and biases
            self.weight_dict['V'] = V
            self.weight_dict['W'] = W
            self.weight_dict['b'] = b
            self.weight_dict['A'] = A
            self.weight_dict['c'] = c
        else:
            self.weight_dict['V'] = np.dstack((self.weight_dict['V'], V))
            self.weight_dict['W'] = np.dstack((self.weight_dict['W'], W))
            self.weight_dict['b'] = np.dstack((self.weight_dict['b'], b))
            self.weight_dict['A'] = np.dstack((self.weight_dict['A'], A))
            self.weight_dict['c'] = np.dstack((self.weight_dict['c'], c))


class TimeHistory(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.perf_counter()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.perf_counter() - self.totaltime
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.perf_counter() - self.epoch_time_start)
        
class RegLoss(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.regloss = []
    # def __init__(self):
    #     super(RegLoss, self).__init__()
    #     self.regloss = []
    
    def on_epoch_end(self, epoch, logs={}):
    ##record weighted matrix
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        ##calculate obj
        self.regloss_k = ObjValue(x_trainset, y_trainset, A, W, V, b, c, lambda1, lambda2, lambda3, lambda4, lambda5, Nx, Nh, Ny, T)
        ##record
        self.regloss.append(self.regloss_k)
    
class TrainErrors(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.trainerr = []
        # Calculate initial error
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        _, _, _, _, self.y_hat = CalculY(x_trainset, A, W, V, b, c, T, Nh, Ny)
        initial_train_err = (1/T) * np.sum((self.y_hat - y_trainset) ** 2)
        self.trainerr.append(initial_train_err)
    
    def on_epoch_end(self, epoch, logs={}):
    ##record weighted matrix
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        ##calculate y_hat
        _, _, _, _, self.y_hat = CalculY(x_trainset, A, W, V, b, c, T, Nh, Ny)
        self.trainerr.append((1/T)*np.sum((self.y_hat-y_trainset)**2))


class TestErrors(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.testerr = []
        # Calculate initial error
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        # _, _, _, _, self.y_hat_test = CalculY(x_testset, A, W, V, b, c, T_test, Nh, Ny)
        _, _, _, _, self.y_hat_test = CalculY(np.concatenate((x_trainset, x_testset), axis=0), A, W, V, b, c, T+T_test, Nh, Ny)
        self.y_hat_test = self.y_hat_test[T:,]
        initial_test_err = (1/T_test) * np.sum((self.y_hat_test - y_testset) ** 2)
        self.testerr.append(initial_test_err)
    
    def on_epoch_end(self, epoch, logs={}):
    ##record weighted matrix
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        ##calculate y_hat
        # _, _, _, _, self.y_hat_test = CalculY(x_testset, A, W, V, b, c, T_test, Nh, Ny)
        _, _, _, _, self.y_hat_test = CalculY(np.concatenate((x_trainset, x_testset), axis=0), A, W, V, b, c, T+T_test, Nh, Ny)
        self.y_hat_test = self.y_hat_test[T:,]
        self.testerr.append((1/T_test)*np.sum((self.y_hat_test-y_testset)**2))  


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_trainset, y_trainset, batchsize, timelength):
        self.x_trainset = x_trainset
        self.y_trainset = y_trainset
        self.batchsize = batchsize
        self.timelength = timelength
        self.num_samples = math.ceil(timelength / batchsize)
        self.epoch_data = []
        
    def __len__(self):
        # 返回每个 epoch 中的批次数量
        return self.num_samples
    
    def __getitem__(self, index):
        # 生成一个批次的数据
        start_indices = np.random.randint(0, len(self.x_trainset) - self.batchsize, self.num_samples)
        batch_x = np.array([self.x_trainset[start:start + self.batchsize] for start in start_indices])
        batch_y = np.array([self.y_trainset[start:start + self.batchsize] for start in start_indices])
        
        # 将 NumPy 数组转换为 TensorFlow 张量
        batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
        
        # 记录这个批次的数据
        self.epoch_data.append((batch_x, batch_y))
        
        return batch_x, batch_y


class PrintEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} complete")
            
#%%Import datasets
np.random.seed(42)
random.seed(42)

## Set dimensions
Nh = 4
Nx = 5
Ny = 3
T_total = 10 #time length

 
SynData = np.loadtxt("SynDataset_Nh%g_Nx%g_Ny%g_T%d_new.txt"  %(Nh, Nx, Ny, T_total), delimiter = ",")
x_dataset = SynData[:, 0:Nx]
y_dataset = SynData[:, Nx:]

#%%Split dataset 
sizerate_training = 0.8
train_length = int(sizerate_training * T_total)

x_trainset = x_dataset[:train_length]
x_testset = x_dataset[train_length:]

y_trainset = y_dataset[:train_length]
y_testset = y_dataset[train_length:]

T = x_trainset.shape[0]
T_test = y_testset.shape[0]

 
lambda1 = 1.2 / (Nh * Ny)  # regularization parameters
lambda2 = 1.2 / (Nh * Nh)
lambda3 = 1.2 / (Nx * Nh)
lambda4 = 1.2 / Nh
lambda5 = 1.2 / Ny 



#%% Define RNN model
timelength = T
timelength_test = T_test

opti_list = np.array(["GD", "GDC", "GDNes", "SGD", "Adam"])
# opti_list = np.array(["GD"]) 
# Initialize the learning rate variable
lr = None

#%% Convert datasets to tensorflow
# Define the new shapes
x_train_shape = [x_trainset.shape[0], 1, x_trainset.shape[1]]
x_test_shape = [x_testset.shape[0], 1, x_testset.shape[1]]
y_train_shape = [y_trainset.shape[0], 1, Ny]
y_test_shape = [y_testset.shape[0], 1, Ny]

# Convert and reshape the datasets
x_input = convert_and_reshape(x_trainset, x_train_shape)
x_test_input = convert_and_reshape(x_testset, x_test_shape)
y_input = convert_and_reshape(y_trainset, y_train_shape)
y_test_input = convert_and_reshape(y_testset, y_test_shape)

#由于Tensorflow 2.x版本和Keras 3.x版本中不支持time_major=True，
#所以需要将形状改为 (batch_size, time_steps, features)
# 而目前的形状是 (time_steps, batch_size, features)
x_input = np.transpose(x_input, (1, 0, 2))  
x_test_input = np.transpose(x_test_input, (1, 0, 2)) 
y_input = np.transpose(y_input, (1, 0, 2))  
y_test_input = np.transpose(y_test_input, (1, 0, 2))  


#%% Define hyperparameters for RNNs 
epoch = 100
# Begin loops
for opt in opti_list:
    mean_i = [0] 
    Errors_GDs_SGDs = pd.DataFrame(columns=['TrainErr' + str(opt), 'TestErr' + str(opt), 'Time' + str(opt)])
    print("====================")
    print(f"Optimize method: {str(opt)}")
    print("====================")
    if opt == 'GD':
        init_method = 'Gaussian'
        stddev_i = 0.1
        batchsize = timelength
        cp_norm = None
        lr = 0.0001
        nes = False
        ini_mach_1 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+1) 
        ini_mach_2 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+2)
        ini_mach_3 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+3)
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr, clipnorm = cp_norm, nesterov = nes)
        
    if opt == 'GDC':
        init_method = 'Gaussian'
        stddev_i = 0.1
        batchsize = timelength
        cp_norm = 0.5
        lr = 0.0001
        nes = False
        ini_mach_1 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+1) 
        ini_mach_2 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+2)
        ini_mach_3 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+3)
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr, clipnorm = cp_norm, nesterov = nes)


    if opt == 'GDNes':
        init_method = 'Glorot'
        stddev_i = 20
        batchsize = timelength
        cp_norm = None
        lr = 0.1
        nes = True
        ini_mach_1 = tf.keras.initializers.GlorotNormal(seed=1234+1) 
        ini_mach_2 = tf.keras.initializers.GlorotNormal(seed=1234+2)
        ini_mach_3 = tf.keras.initializers.GlorotNormal(seed=1234+3)        
        # optimizer = tf.keras.optimizers.SGD(learning_rate = lr, clipnorm = cp_norm, nesterov = nes)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=cp_norm, nesterov=True)



    if opt == 'SGD':
        init_method = 'Gaussian'
        stddev_i = 0.1
        batchsize = 4
        lr = 0.0001
        ini_mach_1 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+1) 
        ini_mach_2 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+2)
        ini_mach_3 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+3)
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr)



    if opt == 'Adam':
        init_method = 'Gaussian'
        stddev_i = 0.01  
        batchsize = 1
        lr = 0.1
        ini_mach_1 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+1) 
        ini_mach_2 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+2)
        ini_mach_3 = tf.keras.initializers.RandomNormal(mean=0, stddev=stddev_i, seed=1234+3)
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    


    #%% Define models   
    rnn_cell = tf.keras.layers.SimpleRNNCell(Nh, activation='relu', use_bias=True,
                                             kernel_initializer = ini_mach_1,
                                             recurrent_initializer = ini_mach_2,
                                             kernel_regularizer = keras.regularizers.l2(lambda3),
                                             recurrent_regularizer = keras.regularizers.l2(lambda2),
                                             bias_regularizer = keras.regularizers.l2(lambda4))
    model = tf.keras.Sequential([
        # tf.keras.layers.RNN(rnn_cell, return_sequences=True, time_major=True),
        tf.keras.layers.RNN(rnn_cell, return_sequences=True),
        tf.keras.layers.Dense(units=Ny, use_bias=True,
                              kernel_initializer = ini_mach_3,
                              kernel_regularizer = keras.regularizers.l2(lambda1),
                              bias_regularizer = keras.regularizers.l2(lambda5))
    ])
    
    model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer= optimizer,
    metrics=[tf.keras.metrics.MeanSquaredError()])
    
    y = model(x_input)

    timec = TimeHistory()
    tae = TrainErrors()
    tee = TestErrors()

    if opt in np.array(['GD','GDC','GDNes']):
        history = model.fit(
            x_input, y_input, epochs=epoch, batch_size=batchsize,
            validation_data=(x_test_input, y_test_input), verbose=0, shuffle=False,
            callbacks=[timec, tae, tee, PrintEpochCallback()])
        
    elif opt in np.array(['SGD']):
        train_gen = CustomDataGenerator(x_trainset, y_trainset, batchsize=4, timelength=timelength)                
        history = model.fit(
            train_gen, epochs=epoch, verbose=0,
            callbacks=[timec, tae, tee, PrintEpochCallback()])
    elif opt in np.array(['Adam']):
        train_gen = CustomDataGenerator(x_trainset, y_trainset, batchsize=1, timelength=timelength)                
        history = model.fit(
            train_gen, epochs=epoch, verbose=0,
            callbacks=[timec, tae, tee, PrintEpochCallback()])
     
    # Record results
    Errors_GDs_SGDs['TrainErr' + str(opt)] = tae.trainerr
    Errors_GDs_SGDs['TestErr' + str(opt)] = tee.testerr
    Errors_GDs_SGDs['Time' + str(opt)] = np.insert(timec.times, 0, 0)

    pd.DataFrame(Errors_GDs_SGDs).to_csv('synT10_Errors_' + str(opt) +'.csv', index_label = "iterations")

    del model
    del rnn_cell
    del history 
    

   

