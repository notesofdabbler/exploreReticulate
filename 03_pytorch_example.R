# pytorch example
# http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-variables-and-autograd
library(reticulate)

use_python("/Users/shanki/anaconda3/bin/python", required = TRUE)
py_config()

torch = import("torch")

main = import_main()
py = import_builtins()

N = 64L
D_in = 1000L
H = 100L
D_out = 10L

dtype = torch$FloatTensor

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
x = torch$randn(N, D_in)$type(dtype)
y = torch$randn(N, D_out)$type(dtype)

# Randomly initialize weights
w1 = torch$randn(D_in, H)$type(dtype)
w2 = torch$randn(H, D_out)$type(dtype)

py_run_string("
def relu(grad_h, h):
    grad_h[h < 0] = 0
    return grad_h
                     ")

learning_rate = 1e-6
for (t in 1:500) {
  h = x$mm(w1)
  h_relu = h$clamp(min=0)
  y_pred = h_relu$mm(w2)
  
  # Compute and print loss
  loss = torch$Tensor$sub(y_pred, y)$pow(2)$sum()
  
  print(paste0(t, ", ", loss))
  
  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = torch$mul(torch$Tensor$sub(y_pred, y), 2.0)

  grad_w2 = h_relu$t()$mm(grad_y_pred)
  grad_h_relu = grad_y_pred$mm(w2$t())
  grad_h = grad_h_relu$clone()
  

  
  grad_h = main$relu(grad_h, h)
  grad_w1 = x$t()$mm(grad_h)
  
  # Update weights using gradient descent
  w1 = torch$Tensor$sub(w1, torch$mul(grad_w1, learning_rate))
  w2 = torch$Tensor$sub(w2, torch$mul(grad_w2, learning_rate))
  
}

