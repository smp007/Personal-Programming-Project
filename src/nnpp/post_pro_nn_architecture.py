"""
====================================================================================================
Script for hyper parameter tuning.
----------------------------------------------------------------------------------------------------
To find out the suitable architechture for the neural network.ie. 
    a)the number of neurons in the 2 hidden layers
Resulting plots are being saved in the `plots` directory
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
from neural_network import*
from symmetry import *
from reader import*
# ==================================================================================================


if __name__ == "__main__":
    #datainput

    path = './dataset_TiO2'
    file_list,energy_list,n = data_read(path)
    energy_list2 = ([(a,b) for a,b in zip(energy_list,n)])
    a,b,c,d = test_train_split(file_list,energy_list,split=99)
    #loading the symmetry function vectors from the corresponding files from which energy value is taken
    test_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) 
    train_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])
    #train set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
    inputs,outputs = zip(*train_xy)
    inputs_ = np.asarray([x.reshape(len(x),1,70) for x in inputs]) #to np arrays
    outputs_ = np.asarray([*outputs])
    data_shuffle(inputs_,outputs_)
    #test set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
    inputs2,outputs2 = zip(*test_xy)
    inputs2_= np.asarray([x.reshape(len(x),1,70) for x in inputs2])
    outputs2_= np.asarray([*outputs2])
    data_shuffle(inputs2_,outputs2_)
    #Min-max normalization
    min_max = np.load('params/min_max_params.npz')
    g_min,g_max = min_max['min'],min_max['max']
    min_max_norm(inputs2_,g_min,g_max)
    min_max_norm(inputs_,g_min,g_max)

    epochs_val = 20
    x_axis = np.linspace(0,epochs_val,epochs_val)   

    fig = plt.figure(figsize = (6,4),dpi =150)
                                                                                
    activations = ['sigmoid','sigmoid','linear']  

    print("----------------------------- Hyperparameter Tuning-----------------------------------\n")
    print('###########    Module to find the suitable architecture for the NN    ##############\n\n')
 
    #Case 1 -- 70-2-2-1 Architecture
    tic = time.time()
    nn_Ti_1 = NeuralNetwork([70,2,2,1],activations)
    nn_O_1  = NeuralNetwork([70,2,2,1],activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_1,nn_O_1,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    toc = time.time()
    plt.plot(x_axis,cost_variation_sgd,'o-k',label='N = 2;'+' t = '+str(round((toc-tic),1))+' sec')
    
    #Case 2 -- 70-5-5-1 Architecture
    tic = time.time()
    nn_Ti_2 = NeuralNetwork([70,5,5,1],activations)
    nn_O_2 = NeuralNetwork([70,5,5,1],activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_2,nn_O_2,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    toc = time.time()
    plt.plot(x_axis,cost_variation_sgd,'p-y',label='N = 5;'+' t = '+str(round((toc-tic),1))+' sec')

    #Case 3 -- 70-7-7-1 Architecture
    tic = time.time()
    nn_Ti_3 = NeuralNetwork([70,7,7,1],activations)
    nn_O_3 = NeuralNetwork([70,7,7,1],activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_3,nn_O_3,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    toc = time.time()
    plt.plot(x_axis,cost_variation_sgd,'*-g',label='N = 7;'+' t = '+str(round((toc-tic),1))+' sec')

    #Case 4 -- 70-11-11-1 Architecture
    tic = time.time()
    nn_Ti_4 = NeuralNetwork([70,11,11,1],activations)
    nn_O_4 = NeuralNetwork([70,11,11,1],activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    toc = time.time()
    plt.plot(x_axis,cost_variation_sgd,'^-b',label='N = 11;'+' t = '+str(round((toc-tic),1))+' sec')

    #Case 5 -- 70-30-30-1 Architecture
    tic = time.time()
    nn_Ti_5 = NeuralNetwork([70,30,30,1],activations)
    nn_O_5 = NeuralNetwork([70,30,30,1],activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_5,nn_O_5,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    toc = time.time()
    plt.plot(x_axis,cost_variation_sgd,'s-r',label='N = 30;'+' t = '+str(round((toc-tic),1))+' sec')
    

    plt.xlabel('epochs')
    plt.ylabel('cost (eV per structure)')
    plt.legend()
    plt.title('Cost variation with different NN architecture (70-N-N-1)')
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/NN architechture.png')

    print('-------------------------------------------------------------------------------------\n')
