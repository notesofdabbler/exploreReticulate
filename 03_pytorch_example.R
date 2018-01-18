library(reticulate)

torch = import("torch")
operator = import("operator")

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

relu = py_run_string("
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
  loss = operator$sub(y_pred, y)$pow(2)$sum()
  
  print(paste0(t, ", ", loss))
  
  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = operator$mul(2, operator$sub(y_pred, y))
  grad_w2 = h_relu$t()$mm(grad_y_pred)
  grad_h_relu = grad_y_pred$mm(w2$t())
  grad_h = grad_h_relu$clone()
  

  
  grad_h = relu$relu(grad_h, h)
  grad_w1 = x$t()$mm(grad_h)
  
  # Update weights using gradient descent
  w1 = operator$sub(w1, operator$mul(learning_rate, grad_w1))
  w2 = operator$sub(w2, operator$mul(learning_rate, grad_w2))
  
}

