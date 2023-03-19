import torch

# create the nodes in the graph, and initialize values
a = torch.tensor(13)
b = torch.tensor(37)

# add together the two values
c = torch.add(a, b)
print(c)

# create the nodes in the graph, and initialize values
a = torch.tensor(2.5)
b = torch.tensor(6.5)

c = torch.add(a, b)
d = torch.subtract(b, 1)
e = torch.multiply(c, d)

print(e)


# simple perceptron with two input nodes
def my_perceptron(x):
    # define some arbitrary weights for the two input values
    W = torch.tensor([[3, -2]], dtype=torch.float32)

    # define the bias of the perceptron
    b = 1

    # compute weighted sum (hint: check out torch.matmul)
    z = totch.matmul(x, W.T) + b

    # apply the sigmoid activation function (hint: use torch.sigmoid)
    output = torch.sigmoid(z)

    return output


sample_input = torch.tensor([[-1, 2]], dtype=torch.float32)

# this should give you a tensor with value 0.002
result = my_perceptron(sample_input)
print(result)


# x: input values
# n_in: number of input nodes
# n_out: number of output nodes
def my_dense_layer(x, n_in, n_out):
    # define variable weights as a matrix and biases
    # initialize weights for one
    # initialize biases for zero
    W = torch.ones((n_in, n_out), requires_grad=True)
    b = torch.zeros((1, n_out), requires_grad=True)

    # compute weighted sum (hint: check out torch.matmul)
    z = torch.matmul(x, W) + b

    # apply the sigmoid activation function (hint: use torch.sigmoid)
    output = torch.sigmoid(z)

    return output


sample_input = torch.tensor([[1, 2.]])
print(my_dense_layer(sample_input, n_in=2, n_out=3))
