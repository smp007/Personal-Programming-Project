"""
====================================================================================================
Script to show the impact of guessed weights in overcoming the local minima issue
----------------------------------------------------------------------------------------------------
This module shows the training epochs with and without guessed weights
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
    f = open('results/result_validation.txt','w') #writes results to txt file
    print("\n-----------------------------    Impact of weights     -------------------------------------\n")
    print('######## Shows the impact of guessed weights in overcoming the local minima issue. #######\n\n')
    print('Data loading...this might take few minutes...\n\n')    
    path = './dataset_TiO2'
    file_list,energy_list,n = data_read(path)
    energy_list2 = ([(a,b) for a,b in zip(energy_list,n)])
    a,b,c,d = test_train_split(file_list,energy_list,split=99)
    #loading the symmetry function vectors from the corresponding files from which energy value is taken
    test_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) 
    train_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

    inputs,outputs = zip(*train_xy)
    inputs_ = np.asarray([x.reshape(len(x),1,70) for x in inputs]) #to np arrays
    outputs_ = np.asarray([*outputs])
    data_shuffle(inputs_,outputs_)

    inputs2,outputs2 = zip(*test_xy)
    inputs2_= np.asarray([x.reshape(len(x),1,70) for x in inputs2])
    outputs2_= np.asarray([*outputs2])
    data_shuffle(inputs2_,outputs2_)



    min_max = np.load('params/min_max_params.npz')
    g_min,g_max = min_max['min'],min_max['max']
    min_max_norm(inputs2_,g_min,g_max)
    min_max_norm(inputs_,g_min,g_max)

    epochs_val = 15
    x_axis = np.linspace(0,epochs_val,epochs_val)
 
    fig = plt.figure(figsize = (6,4),dpi =150)

    #loading the NN with guessed weights
    trained_params = np.asarray(load_params('guess'))
    node_list = [70,11,11,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']
    nn_Ti_2 = NeuralNetwork(node_list,activations)
    initialize_params(nn_Ti_2,*trained_params[0:6])
    nn_O_2 = NeuralNetwork(node_list,activations)
    initialize_params(nn_O_2,*trained_params[6:])

    cost_variation_mom,lr_mom = SGD_momentum(nn_Ti_2,nn_O_2,inputs_,outputs_,learning_rate=1e-9,epochs=epochs_val,beta=0.99999) 
    plt.plot(x_axis,cost_variation_mom,label='guessed')

    #NN with random weights
    node_list = [70,11,11,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']
    nn_Ti_3 = NeuralNetwork(node_list,activations)
    nn_O_3 = NeuralNetwork(node_list,activations)

    cost_variation_mom,lr_mom = SGD_momentum(nn_Ti_3,nn_O_3,inputs_,outputs_,learning_rate=1e-6,epochs=epochs_val,beta=0.99999) 
    plt.plot(x_axis,cost_variation_mom,label='random')

    plt.xlabel('epochs')
    plt.ylabel('cost (eV per structure)')
    plt.legend()
    plt.title('Cost variation with diff weights')
    plt.show()
    fig.tight_layout()
    fig.savefig('plots/weights.png')

