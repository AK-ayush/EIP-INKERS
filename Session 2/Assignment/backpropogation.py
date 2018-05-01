import numpy as np 
print (np.__version__)

lr = 1
epoch = 1
def sigmoid(x):
	return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
	return x*(1.0-x)

X = np.array([[1.0, 0.0, 1.0, 0.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0]]) #[3x4]
Y = np.array([[1], [1], [0]]) #[3x1]
print ("X:\n", X, "\n----\n","Y:\n", Y)

'''Hidden layer parameters'''
wh = np.random.random((4,3)) #[4x3]
bh = np.random.random((1,3)) #[1x3]
# wh = np.array([[0.42, 0.88, 0.55], [0.10, 0.73, 0.68], [0.60, 0.18, 0.47], [0.92, 0.11, 0.52]])
# bh = np.array([0.46, 0.72, 0.08])
print ("wh:\n",wh, "\n----\n", "bh:\n",bh, "\n----")

'''output layer parameters'''
wout = np.random.random((3,1)) #[3x1]
bout = np.random.random((1)) #[1]
# wout = np.array([[0.30], [0.25], [0.23]])
# bout = [0.69]
print ("wout:\n", wout, "\n----\n", "bout:\n", bout, "\n----")

for e in range(epoch):
	print("-----------------------------\nepoch:",e+1,"\n-----------------------------")
	'''hidden layer computations'''
	hidden_layer_input = np.matmul(X,wh) + bh #[3x3]
	print("hidden_layer_input:\n", hidden_layer_input, "\n-----")

	hidden_layer_activations = sigmoid(hidden_layer_input)
	print("hidden_layer_activations\n", hidden_layer_activations, "\n-----")

	'''output layer computations'''
	output_layer_input = np.matmul(hidden_layer_activations, wout)+bout
	print("output_layer_input:\n",output_layer_input, "\n-----")

	output = sigmoid(output_layer_input)
	print("output: \n", output, "\n-----")
	# print(output.shape, "\n-----")
	# print(Y, "\n-----")
	'''computing error at final layer'''
	E = np.subtract(Y,output) #[3]
	print("Error\n",E, "\n-----")

	'''caculate the slope at output and hidden layer'''
	slope_output_layer = derivative_sigmoid(output)
	print("slope_output_layer:\n",slope_output_layer, "\n-----")

	slope_hidden_layer = derivative_sigmoid(hidden_layer_activations)
	print("slope_hidden_layer:\n",slope_hidden_layer, "\n-----")

	'''computing delta_output'''
	d_output = E*slope_output_layer*lr
	print("d_output:\n",d_output, "\n-----")


	'''computing delta_hidden layer'''
	error_at_hidden_layer = np.matmul(d_output, np.transpose(wout))
	print("error_at_hidden_layer:\n",error_at_hidden_layer, "\n-----")

	d_hidden_layer = error_at_hidden_layer*slope_hidden_layer
	print("d_hidden_layer:\n",d_hidden_layer, "\n-----")

	'''Update weights at both output and hdden layer'''
	wout = wout + np.matmul(np.transpose(hidden_layer_activations), d_output)*lr
	print("wout: \n", wout, "\n-----")

	wh = wh + np.matmul(np.transpose(X), d_hidden_layer)*lr
	print("wh:\n", wh, "\n-----")

	'''Update biases at both output and hidden layer'''
	bout = bout + np.sum(d_output, axis=0)*lr
	print("bout:\n", bout, "\n-----")

	bh = bh + np.sum(d_hidden_layer, axis=0)*lr
	print("bh:\n", bh, "\n-----")

	# print (X.shape, Y.shape, wh.shape, bh.shape, hidden_layer_input.shape)
