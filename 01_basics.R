

library(reticulate)

main = import_main()
py = import_builtins()

np = import("numpy")

os = import("os")
os$getcwd()
os$chdir("/Users")

getwd()
setwd("~/Documents/notesofdabbler/exploreReticulate/")
os$getcwd()

x = "a b c"
r_to_py(x)$split()
class(r_to_py(x)$split())

py_to_r(r_to_py(x)$split())
class(py_to_r(r_to_py(x)$split()))


np$arange(10)
x = np$cumsum(np$arange(10))
class(x)

x = matrix(seq(1,12), nrow = 4)
x
np_array(x)
class(np_array(x))

x = np_array(seq(1,12), c(3,4))
x = np_array(seq(1,12), c(3,4), order = "C")

iterate(py$iter(c(1,2,3,4)), print)
iterate(py$iter('abcd'), print)

x = py$iter(c(1,2,3,4))
while (TRUE) {
  item = iter_next(x)
  print(item)
  if(is.null(item)) {
    break
  }
}

x = py$range(1000L)
while (TRUE)  {
  val = iter_next(x)
  print(val)
  if (val > 100) {
    break
  }
}

mypyplots = import_from_path("mypyplots", path = "./")

x = seq(0, 10, length.out = 500)
dashes = c(10, 5, 100, 5)
mypyplots$plot1(x, dashes)

x = seq(-5, 5, 0.25)
y = seq(-5, 5, 0.25)

XYgrid = np$meshgrid(x, y)

zfun = function(x, y) { sin(sqrt(x**2 + y**2)) }
Z = outer(x, y, FUN = zfun)


mypyplots$pyplot3d(np_array(XYgrid[[1]]), np_array(XYgrid[[2]]), np_array(Z))

mypyplots$pyplot3d()
