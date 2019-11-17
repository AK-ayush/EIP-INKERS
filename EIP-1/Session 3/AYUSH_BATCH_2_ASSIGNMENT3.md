[comment]: <> (FIRSTNAME_BATCH_X_ASSIGNMENT3)

| ![VideoBlocks](https://avatars0.githubusercontent.com/u/11587734?s=400&u=999125efb5b5a9abd271b85e55ffcb230e1253a4&v=4 =200x200)  | Ayush Kumar <br/> Batch 2 <br/> kayush206@gmail.com|
|:---:|:---:|

___
### 1. Cross-Entropy Loss:

In machine learning, loss functions are used to measure the performances of model. In order to maximize the model performance, loss functions are required to minimized. An ideal machine learning model will have zero loss. 
Cross-Entropy Loss or Log Loss is being largely used in classification problems where the predicted output is between 0 and 1 as softmax is used in last layer. Cross-Entropy loss decreases logrithmically as the predicted probability converges toward the actual label. So predicting a probability of 0.86 when actual label is 1 would be good and result in a low value of loss. An ideal classification model would have a cross-entropy loss of 0.

![cross_entorpy](http://ml-cheatsheet.readthedocs.io/en/latest/_images/cross_entropy.png =400x300)

In the above graph, range of possible values of log-loss have been shown, given a true observation (isElephant = 1). As the predicted output approaches to 1, cross-entropy loss decreases slowly. Whereas, the cross-entropy increases rapidaly, as predicted output decreases. Log-loss penalizes perticularaly those predictions which are confident and wrong.

#### Cross-Entropy Loss =:
Binary Classification $=-((1-y_{true})ln(1-y_{pred})+ y_{true}ln(y_{pred}))$

Multi-Class Classification $=-\sum _{c=1} ^{C}y_{true,c}ln(y_{pred,c})$ 

#### Examples:

`y_true = [ 0.0, 1.0, 0.0]`
`y_pred = [0.228, 0.619, 0.153]`
`E = - (0.0*ln(0.228) + 1.0*ln(0.619) + 0.0*ln(0.153))`
`E = 0.479`

___
### 2. Dropout:
Dropout is one of the most important techniques which are being used actively to combat overfitting in deep neural network. Dropout refers to dropping out units in both hidden and visible layer in a neural network. In simple words, dropout means to ignoring a certain set of nuerons which is chosen at random during the training phase. These ignored neurons are not considered during a perticular forward and backward pass.
In more technical terms, In each training stage, some randomly chosen nodes are droped out of the neural network with probability $1-p$ or we can say, some randomly chosen nodes are kept in the neural network with probability $p$,  which resulted into a reduced network and incoming and outcoming connections to these dropped-out nodes are eleminated. 

Sole purpose of using dropout to **"prevent over-fiting"**. a densely connected layer contains most of the parameters, and hence, neurons learn co-dependecy among each other during training which restrained the individual capacity of each neuron resulting to over-fitting.

![drop_out](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/dropout.png)

#### Important facts:

[1] Helps neural network to learn more robust features of training data
[2] Reduces the training time for each epoch but doubles the number of epochs required to converge roughly.
[3] With N hidden units, $2^N$ possible models were considered during training phase since each of the hidden units can be dropped. While testing phase, each node is considered in network and reduction in each activation by a factor of $p$.  

___
### 3. ReLU (Rectifier Linear Unit):

Rectifier Linear Unit (ReLU) is most frequently use activation function in deep neural networks. It has become very much popular in recent years. Many recent state of the art models used ReLU for hidden layer activation. ReLU is the simplest activation function you can use. This function returns input itself, if input is positive and penalize the negative input and returns 0.
$$f(x) = max(0,x)$$
![relu](https://i.imgur.com/gKA4kA9.jpg)

ReLU is non-saturating non-linear activation while previously dominated activations were saturating such sigmoid, tanh and etc. Compared to sigmoid/tanh, ReLU reduces the training steps required for SGD to converge because of non-saturating ability. It can be implemented by just thresholding an array of activation at 0 while the tanh/sigmoid require expensive mathematical operations i.e. exponantials. When the input is greter than zero then derivative is one, so there won't be gradient squeezing scenario which is normal in sigmoid. ReLU is most favourable activation function in order to train very deep neural nets.

![relu_trainfast](http://cs231n.github.io/assets/nn1/alexplot.jpeg)
 Source: [Krizhevsky et al.](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) paper indicating the 6x improvement in convergence with the ReLU unit compared to the tanh unit.
But there are some drawbacks of using this simplest activation function, From [cs231n Course:](http://cs231n.github.io/neural-networks-1/#nn)
_"Unfortunately, ReLU units can be fragile during training and can "die". For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be "dead" (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue"._
___
References:
-
1) https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5
2) http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
3) http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
4) https://en.wikipedia.org/wiki/Loss_functions_for_classification
5) https://en.m.wikipedia.org/wiki/Cross_entropy
6) http://cs231n.github.io/neural-networks-1/#actfun
7) https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning