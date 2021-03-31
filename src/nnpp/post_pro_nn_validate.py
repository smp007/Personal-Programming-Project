"""
====================================================================================================
Scripts for checking the sanity of neural network predictions.
----------------------------------------------------------------------------------------------------
This module,
-Uses the trained neural network to predict the energy for the test dataset and generates 3 plots
    a)Predicted value vs Reference value
    b)Error in prediction
    c)ANN vs DFT
-Determines the correlation coefficient
====================================================================================================

"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
from neural_network import*
from symmetry import *
from reader import*
# ==================================================================================================

#Extracting the weights of trained NN
trained_params = np.asarray(load_params('trained'))
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']
nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[0:6])
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])

if __name__ == "__main__":
    f = open('results/result_validation.txt','w') #writes results to txt file
    print("\n-----------------------------     NN-Validation      -------------------------------------\n")
    print("\n-----------------------------     NN-Validation      -------------------------------------\n",file=f)
    print('########  Validates the NN with predictions on test set,finds the correlation coeff. #######\n\n')
    print('Data loading...this might take few minutes...\n\n')    
    path = './dataset_TiO2'
    file_list,energy_list,n = data_read(path)
    energy_list2 = ([(a,b) for a,b in zip(energy_list,n)])
    a,b,c,d = test_train_split(file_list,energy_list,split=99)
    #loading the symmetry function vectors from the corresponding files from which energy value is taken
    test_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) 
    train_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

    #trainset
    inputs,outputs = zip(*train_xy)
    inputs_ = np.asarray([x.reshape(len(x),1,70) for x in inputs]) #to np arrays
    outputs_ = np.asarray([*outputs])
    data_shuffle(inputs_,outputs_)

    #testset
    inputs2,outputs2 = zip(*test_xy)
    inputs2_= np.asarray([x.reshape(len(x),1,70) for x in inputs2])
    outputs2_= np.asarray([*outputs2])
    data_shuffle(inputs2_,outputs2_)


    #normalization
    min_max = np.load('params/min_max_params.npz')
    g_min,g_max = min_max['min'],min_max['max']
    min_max_norm(inputs2_,g_min,g_max)
    min_max_norm(inputs_,g_min,g_max)
 
    #prediction
    x,y,z = predict_energy(inputs2_,outputs2_,nn_Ti,nn_O)

    #plots prediction v reference for test set
    fig = plt.figure(figsize = (7,4),dpi =150)
    plt.plot(y,'o:y',label='reference')
    plt.plot(x,'.--r',label='predicted')
    plt.xlabel('m')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.title('Predicted v Reference')
    plt.show()
    fig.tight_layout()
    fig.savefig('plots/predict_v_reference_test_set.png')

    #plots error in predictions of test set
    fig = plt.figure(figsize = (7,4),dpi =150)
    plt.plot(z,'.:b')
    plt.xlabel('m')
    plt.ylabel('Error (eV per atom)')
    plt.title('Test set error')
    fig.tight_layout()
    plt.grid('True')    
    plt.show()
    fig.savefig('plots/testset_error.png')

    #plots ANN prediction against DFT reference values
    fig, ax = plt.subplots()
    plt.plot(y,x,'o:k')
    ax.set_xlabel('ANN energy (eV)')
    ax.set_ylabel('DFT energy (eV)')
    ax.set_title('ANN v DFT')
    plt.grid('True')     
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/ANN_v_DFT.png')

    a,b,_ = predict_energy(inputs_,outputs_,nn_Ti,nn_O)
    r_squared = correlation_coefficient(a,b)
    c,d,e = predict_energy(inputs2_,outputs2_,nn_Ti,nn_O)
    q_squared = correlation_coefficient(c,d)
    #prints result to screen
    print('\n##################                  RESULTS                  ########################\n')
    print('{0: <25}'.format('MSE of test set(eV/atom)'),'=',0.5*np.mean(np.square(e)),'\n')
    print('{0: <25}'.format('R squared value'),'=',r_squared,'\n')
    print('{0: <25}'.format('Q squared value'),'=',q_squared,'\n')
    print('{0: <25}'.format('Correlation coefficient'),'=',r_squared/q_squared,'\n')
    print('########################################################################################\n')
    #prints result to file
    print('\n##################                  RESULTS                  ########################\n',file=f)
    print('{0: <25}'.format('MSE of test set(eV/atom)'),'=',0.5*np.mean(np.square(e)),'\n',file=f)
    print('{0: <25}'.format('R squared value'),'=',r_squared,'\n',file=f)
    print('{0: <25}'.format('Q squared value'),'=',q_squared,'\n',file=f)
    print('{0: <25}'.format('Correlation coefficient'),'=',r_squared/q_squared,'\n',file=f)
    print('########################################################################################\n',file=f)


