
# Assignment-1

Accuracy of the model is `99.13%`.

```python
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

[0.032263540528796875, 0.9913]
```
[Code](./Assignment-mnist.ipynb)
---
---
---

Convolution:
-----------
Convolutional is a matrix dot product and then summation operation, which is being applied on a input matrix to automate the feature engineering process. In convolutional layer, it's like a flashlight is being slided on each pixel of matrix and this flashlight can be understood as kernel/filter. A convolutional layer outputs a feature map containing the learnt features by this operation.

Filters/Kernels:
---------------
A filter or kernel is a matrix of variables which are learned using the backpropagation algorithm. Each kernel extracts a very specific kind of feature or pattern when convolved across the input image. Number of channels in each filter must be equal to number of channels in input image and number of filters determines the number of channels in output feature map.

Epochs:
------
An epoch is said be completed when entire training data is passed in forward propogation and performed the back propogation and updated the weights throught neural networks only once.


1x1 Convolution:
---------------
1x1 convolution kernel/filter is being applied when you want to alter the number of channels in input matrix. It is mostly used to reduce the number of channels but sometimes is used otherwise too.

3x3 Convolution:
---------------
It is most widely used convolutional kernel. It can be used repeatedly in order to imitates any arbitary shaped kernel with less number of parameters.(i.e. 3x3 can be used 2 times to imitates 5x5 with 18 parametes only as opposed to 25 parameters.) Also, it has been very well optimized for specific devices such GPUs, raspberrypi and etc.


Feature Maps:
------------
When we apply a convolutional layer followed by some activation function on a input images the output it generates is called feature maps.


Receptive Field:
---------------
There are two kind of receptive fields. In a layer, the number of pixels in the previous layer that can be seen from a single pixels of feature map of current layer, is called `Local Receptive Field`, while the number of pixels in original input image that can be seen from a single pixels of feature map of current layer is called the `Global Receptive Field`. And the number of required layers in a neural network can be determined by Global receptive fields in such way that global receptive field of last layer should be at least the number of pixels in object in the image.  


Activation Function:
-------------------
activation fuctions are the non-linear funtions which are being applied after each layers in neural networks. They are necessary, so that the network can learn both linear and non-linear functions since neural network are supposed to approximate all the functions.



