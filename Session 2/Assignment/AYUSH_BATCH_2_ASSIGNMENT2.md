[comment]: <> (FIRSTNAME_BATCH_X_ASSIGNMENT2)

| ![VideoBlocks](https://avatars0.githubusercontent.com/u/11587734?s=400&u=999125efb5b5a9abd271b85e55ffcb230e1253a4&v=4 =200x200)  | Ayush Kumar <br/> Batch 2 <br/> kayush206@gmail.com|
|:---:|:---:|

___
Part A :
-
Here is [github](https://github.com/AK-ayush/EIP-INKERS/blob/master/Session%202/Assignment/python_numpy_tutorial.ipynb) link where ipython notebook is uploaded after updating the all variables accordingly in the existing notebook.

___

Part B :
-
I have written a python code to simulate the backpropogation using only numpy which can be found [here](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/backpropogation.py)

**Step 0:** Read input and output:

`X = np.array([[1.0, 0.0, 1.0, 0.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0]]) #[3x4]`
`Y = np.array([[1], [1], [0]]) #[3x1]`

![step_0](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_1.png) 

<br/>

**Step 1:** Initialize weights and biases with random values (There are methods to initialize weights and biases but for now initialize with random values using python numpy scientific library)
`wout = np.random.random((3,1)) #[3x1]`
`bout = np.random.random((1)) #[1]`

![step_1](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_2.png) 

<br/>

**Step 2:** Calculate hidden layer input:
`hidden_layer_input = np.matmul(X,wh) + bh #[3x3]`

![step_2](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_3.png) 

<br/>

**Step 3:** Perform non-linear transformation on hidden linear input, in this case we are using *sigmoid activation function*.
`hiddenlayer_activations = sigmoid(hidden_layer_input) #[3x3]`

![step_3](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_4.png)

<br/>

**Step 4:** Perform linear and non-linear transformation of hidden layer activation at output layer and compute error(E):
`output_layer_input = np.matmul(hidden_layer_activations, wout)+bout #[3x1]`
`output = sigmoid(output_layer_input) #[3x1]`

`E = np.subtract(Y,output) #[3x1]`

![step_4](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_5.png)

<br/>

**Step 5:** Compute slope at output and hidden layer
`slope_output_layer = derivative_sigmoid(output)`
`slope_hidden_layer = derivative_sigmoid(hidden_layer_activations)`

![step_5](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_6.png)

<br/>

**Step 7:** Compute delta at output layer:
`d_output = E*slope_output_layer*lr #lr=1 learning rate`

![step_6](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_7.png)

<br/>

**Step 8:**  Calculate Error at hidden layer:
`error_at_hidden_layer = np.matmul(d_output, np.transpose(wout))`
`d_hidden_layer = error_at_hidden_layer*slope_hidden_layer`

![step_7](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_8.png)

<br/>

**Step 9:** Update weight at both output and hidden layer:
`wout = wout + np.matmul(np.transpose(hidden_layer_activations), d_output)*lr`

`wh = wh + np.matmul(np.transpose(X), d_hidden_layer)*lr`

![step_8](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_9.png)

<br/>

**Step 10:** Update biases at both output and hidden layer:
`bout = bout + np.sum(d_output, axis=0)*lr`

`bh = bh + np.sum(d_hidden_layer, axis=0)*lr`

![step_9](https://github.com/AK-ayush/EIP-INKERS/raw/master/Session%202/Assignment/images/Session2_10.png)

