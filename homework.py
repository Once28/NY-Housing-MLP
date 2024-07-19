from mlp import LinearLayer, ReLU, LeakyReLU
from utils import softmax_cross_entropy, data_loader, predict_label, DataSplit
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import argparse
import json
import time
import pdb




### Model forward pass ###
def forward_pass(model, x, y):
    a1 = model['L1'].forward(x)  # output of the first linear layer
    # print(f"Output of L1 (a1) dimensions: {a1.shape}")
    h1 = model['nonlinear1'].forward(a1)  # output of the first ReLU
    # print(f"Output of ReLU1 (h1) dimensions: {h1.shape}")
    a2 = model['L2'].forward(h1)  # output of the second linear layer
    # print(f"Output of L2 (a2) dimensions: {a2.shape}")
    h2 = model['nonlinear2'].forward(a2)  # output of the second ReLU
    # print(f"Output of ReLU2 (h2) dimensions: {h2.shape}")
    a3 = model['L3'].forward(h2)  # output of the third linear layer
    # print(f"Output of L3 (a3) dimensions: {a3.shape}")
    h3 = model['nonlinear3'].forward(a3)  # output of the third ReLU
    # print(f"Output of ReLU2 (h2) dimensions: {h2.shape}")
    a4 = model['L4'].forward(h3)  # Output of the fourth linear layer
    # print(f"Output of L4 (a4) dimensions: {a4.shape}")
    loss = model['loss'].forward(a4, y)  # Computing loss using output of L4
    # print(f"Output of Loss dimensions: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")

    return a1, h1, a2, h2, a3, h3, a4, loss


def backward_pass(model, x, a1, h1, a2, h2, a3, h3, a4, y):
    grad_a4 = model['loss'].backward(a4, y)
    grad_h3 = model['L4'].backward(h3, grad_a4)
    grad_a3 = model['nonlinear3'].backward(a3, grad_h3)
    grad_h2 = model['L3'].backward(h2, grad_a3)
    grad_a2 = model['nonlinear2'].backward(a2, grad_h2)
    grad_h1 = model['L2'].backward(h1, grad_a2)
    grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
    grad_x = model['L1'].backward(x, grad_a1)
    return grad_x


### Compute the accuracy and loss of a model on some train/val/test dataset ###
def compute_accuracy_loss(N_data, DataSet, model, minibatch_size=100):
    acc = 0.0
    loss = 0.0
    count = 0

    for i in range(int(np.floor(N_data / minibatch_size))):
        x, y = DataSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

        _, _, _, _, _, _, a4, batch_loss = forward_pass(model, x, y)
        loss += batch_loss
        acc += np.sum(predict_label(a4) == y)
        count += len(y)

    return acc / count, loss


def miniBatchGradientDescent(model, _learning_rate):
    lambda_reg = 0.001
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                g = module.gradient[key]
                g_numeric = g.astype(float)
                g_numeric += 2 * lambda_reg * module.params[key]
                module.params[key] -= _learning_rate * g_numeric
    return model


def magnitude_checker(DataSet, model):
    x, y = DataSet.get_example(np.arange(50))
    a1, h1, a2, h2, a3, h3, a4, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, h2, a3, y)
    l1_norm_w_grad_five = model["L1"].gradient["W"].sum()
    x, y = DataSet.get_example(np.arange(5000))
    a1, h1, a2, h2, a3, h3, a4, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, h2, a3, y)
    l1_norm_w_grad_fivek = model["L1"].gradient["W"].sum()
    print(
        "Check the magnitude (L1-norm of layer L1) of gradient with batch size 50: {:.6f} and with batch size 5k: {:.6f}".format(
            l1_norm_w_grad_five, l1_norm_w_grad_fivek))


def gradient_checker(DataSet, model):
    x, y = DataSet.get_example([0])

    a1, h1, a2, h2, a3, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, h2, a3, y)

    grad_dict = {}
    for layer in ['L1', 'L2', 'L3', 'L4']:
        grad_dict[f"{layer}_W_grad_first_dim"] = model[layer].gradient["W"][0][0]
        grad_dict[f"{layer}_b_grad_first_dim"] = model[layer].gradient["b"][0]

    for name, grad in grad_dict.items():
        layer_name, param_type, _ = name.split("_")
        epsilon_value = 1e-3
        original_value = model[layer_name].params[param_type]
        epsilon_vector = np.zeros_like(original_value)
        np.put_along_axis(epsilon_vector, np.array([[0]]), epsilon_value, axis=1 if param_type == 'W' else 0)

        model[layer_name].params[param_type] = original_value + epsilon_vector
        f_w_add_epsilon = forward_pass(model, x, y)[-1]

        model[layer_name].params[param_type] = original_value - epsilon_vector
        f_w_sub_epsilon = forward_pass(model, x, y)[-1]

        approximate_gradient = (f_w_add_epsilon - f_w_sub_epsilon) / (2 * epsilon_value)

        model[layer_name].params[param_type] = original_value

        print(
            f"Check the gradient of {param_type} in the {layer_name} layer from backpropagation: {grad:.6f} and from approximation: {approximate_gradient:.6f}")


max_grad_norm = 1.0  # Adjust this value as needed


def clip_gradients(model):
    for module_name, module in model.items():
        if hasattr(module, 'gradient'):
            for key, _ in module.gradient.items():
                grad_norm = np.linalg.norm(module.gradient[key])
                if grad_norm > max_grad_norm:
                    module.gradient[key] *= max_grad_norm / grad_norm
    return model


def get_learning_rate(initial_learning_rate, decay_rate, decay_steps, t):
    return initial_learning_rate * decay_rate ** (t // decay_steps)


def main(main_params):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, num_classes = data_loader()
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape
    N_test, _ = Xtest.shape
    print("Training data size: %d, Validation data size: %d, Test data size: %d" % (N_train, N_val, N_test))

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)
    testSet = DataSplit(Xtest, Ytest)

    model = dict()
    num_L1 = 512
    num_L2 = 256
    num_L3 = 256
    num_L4 = num_classes

    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])
    check_gradient = main_params['check_gradient']
    check_magnitude = main_params['check_magnitude']
    patience = int(main_params['early_stopping_patience'])

    train_acc_record = []
    train_loss_record = []
    val_acc_record = []
    val_loss_record = []
    best_epoch = 0
    best_model = None

    _learning_rate = float(main_params['learning_rate'])
    initial_learning_rate = _learning_rate
    decay_rate = 0.1
    decay_steps = 10

    model['L1'] = LinearLayer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = LeakyReLU()
    model['L2'] = LinearLayer(input_D=num_L1, output_D=num_L2)
    model['nonlinear2'] = LeakyReLU()
    model['L3'] = LinearLayer(input_D=num_L2, output_D=num_L3)
    model['nonlinear3'] = LeakyReLU()
    model['L4'] = LinearLayer(input_D=num_L3, output_D=num_L4)
    model['loss'] = softmax_cross_entropy(num_classes=num_L4)

    if check_magnitude:
        magnitude_checker(trainSet, model)

    if check_gradient:
        gradient_checker(trainSet, model)

    start_time = time.time()

    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))

        _learning_rate = get_learning_rate(initial_learning_rate, decay_rate, decay_steps, t)

        idx_order = np.random.permutation(N_train)

        for i in tqdm(range(int(np.floor(N_train / minibatch_size)))):
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            a1, h1, a2, h2, a3, h3, a4, _ = forward_pass(model, x, y)

            backward_pass(model, x, a1, h1, a2, h2, a3, h3, a4, y)
            model = clip_gradients(model)
            # Pass the current epoch (t) to the miniBatchGradientDescent function
            model = miniBatchGradientDescent(model, _learning_rate)

        train_acc, train_loss = compute_accuracy_loss(N_train, trainSet, model)
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        val_acc, val_loss = compute_accuracy_loss(N_val, valSet, model)
        val_acc_record.append(val_acc)
        val_loss_record.append(val_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))
        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

        if val_acc == max(val_acc_record):
            best_model = deepcopy(model)
            best_epoch = t + 1
            patience = int(main_params['early_stopping_patience'])
        else:
            patience -= 1

        if patience == 0:
            break

    end_time = time.time()

    test_acc, test_loss = compute_accuracy_loss(N_test, testSet, best_model)
    print('Test accuracy at the best epoch (epoch ' + str(best_epoch) + ') is ' + str(test_acc))

    json.dump({'train': train_acc_record, 'val': val_acc_record, 'test': test_acc, 'time': end_time - start_time},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_b' + str(main_params['minibatch_size']) +
                   '.json', 'w'))

    print('Training time: ' + str(end_time - start_time))
    print('Finish running!')
    return train_loss_record, val_loss_record


def get_parser():
    ######################################################################################
    # These are the default arguments used to run your code.
    # You can modify them to test your code
    ######################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--minibatch_size', default=100)
    parser.add_argument('--early_stopping_patience', default=3)
    parser.add_argument('--check_gradient', action="store_true", default=False,
                        help="Check the correctness of the gradient")
    parser.add_argument('--check_magnitude', action="store_true", default=False,
                        help="Check the magnitude of the gradient")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main_params = get_parser()
    main(main_params)