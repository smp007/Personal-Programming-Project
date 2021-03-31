"""
====================================================================================================
Script for hyper parameter tuning.
----------------------------------------------------------------------------------------------------
To find out the suitable optimiser for the prediction of the potential.Extracts the cost variation
for the neural network with different optimizers and plots the results.
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

    epochs_val = 30
    x_axis = np.linspace(0,epochs_val,epochs_val)

    fig = plt.figure(figsize = (6,4),dpi =150)

    node_list = [70,11,11,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']  

    print("----------------------------- Hyper parameter tuning ----------------------------------\n")
    print('###########  Module to find the most suitable optimizer for the training.  ##############\n\n')
    #Minibatch gradient descent
    nn_Ti_1 = NeuralNetwork(node_list,activations)
    nn_O_1  = NeuralNetwork(node_list,activations)
    cost_variation_mbgd,lr_mbgd = minibatch_gradient_descent(nn_Ti_1,nn_O_1,inputs_,outputs_,learning_rate=1e-6,batchSize=50,epochs=epochs_val)
    plt.plot(x_axis,cost_variation_mbgd,'o-k',label='minibatch GD; lr=1e-5;  batchsize=50')

    #SGD with momentum
    nn_Ti_2 = NeuralNetwork(node_list,activations)
    nn_O_2 = NeuralNetwork(node_list,activations)
    cost_variation_mom,lr_mom = SGD_momentum(nn_Ti_2,nn_O_2,inputs_,outputs_,learning_rate=1e-6,epochs=epochs_val,beta=0.99999) 
    plt.plot(x_axis,cost_variation_mom,'.:y',label='momentum;  lr=1e-6; '+' $\\beta1$=0.99999')

    #RMSProp
    nn_Ti_3 = NeuralNetwork(node_list,activations)
    nn_O_3 = NeuralNetwork(node_list,activations)
    cost_variation_rms,lr_rms = RMSprop(nn_Ti_3,nn_O_3,inputs_,outputs_,learning_rate=1e-3,epochs=epochs_val,beta=0.9)                       
    plt.plot(x_axis,cost_variation_rms,'*-g',label='RMSprop;  lr=1e-3;'+'  $\\beta1$=0.9')

    #SGD
    nn_Ti_4 = NeuralNetwork(node_list,activations)
    nn_O_4 = NeuralNetwork(node_list,activations)
    cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=5e-8,epochs=epochs_val)
    plt.plot(x_axis,cost_variation_sgd,'.-b',label='SGD;  lr=5e-8')

    #Adam 
    nn_Ti_5 = NeuralNetwork(node_list,activations)
    nn_O_5 = NeuralNetwork(node_list,activations)
    cost_variation_adam,lr_adam = Adam(nn_Ti_5,nn_O_5,inputs_,outputs_,learning_rate=1e-2,epochs=epochs_val,beta1=0.9,beta2=0.999)
    plt.plot(x_axis,cost_variation_adam,'s-r',label='Adam;   lr=1e-2;'+'  $\\beta1$=0.9;'+'  $\\beta2$=0.999')
        

    plt.xlabel('epochs')
    plt.ylabel('cost (eV per structure)')
    plt.legend()
    plt.title('Cost variation with different optimizers')
    plt.show()
    fig.tight_layout()
    fig.savefig('plots/without_guessed_weights__eV_per_structure.png')
    print('------------------------------------------------------------------------------------------\n')
