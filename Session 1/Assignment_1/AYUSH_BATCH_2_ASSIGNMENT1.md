[comment]: <> (FIRSTNAME_BATCH_X_ASSIGNMENT1)

| ![avatar](https://avatars0.githubusercontent.com/u/11587734?s=400&u=999125efb5b5a9abd271b85e55ffcb230e1253a4&v=4 =200x200)  | Ayush Kumar <br/> Batch 2 <br/> kayush206@gmail.com|
|:---:|:---:|

<br/>
<br/>
<br/>

In this article, __Convolution__ and __1x1 Convulution__ are the topics which are chosen to elaborated briefly as per the understanding about the domain. And 10 examples of uses of __Mathjax__ is also provided in this. 
___

1] Convolution:
------------
In mathematics, convolution is a binary operator which being applied on two functions $f$ and $g$ to produce third function.
$$(f*g)(c) = \sum_{a} f(a).g(c-a)$$
$$or$$
$$(f*g)(c) = \int_{-\infty}^{\infty}f(a).g(c-a)dc$$

Convolution is follows commutative and associative rules of algebra.
- **Commutativity**
   $$f*g = g*f$$
- **Associativity**
   $$f*(g*h) = (f*g)*h$$

Below visulization depicts the convolution on two box function from wikipedia:

![from_wikipedia](https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif)

It is one of most important operations in **computer vision**, **image** and **signal processing**. It can be applied for speech processing(1D convolution), for image processing(2D convolution)  and video processing (3D convolution). In this article, convolution in 2D spatial is discussed, which is most widely used in image processing as feature extracter and is a fundamental building block of Convolutional Neural Network (CNN). An image in computer is represented by a multi-dimensional array whose elements are numbers between 0 to 255. The size of this multi-dimensional array is Height x Width x #Channels i.e. color image has 3(RGB) channel where a grayscale image has only one channel.


![convolution](http://intellabs.github.io/RiverTrail/tutorial/images/convolution2.png)

Since convolution is a binary operator, it has a kernel(or filter) matrix which can not be grater than the original image in height and width. Sometimes the kernels are pre-learnt for a perticular task, such as blurring, sharpening, edge-detection and etc.

Below there are two example provided in order to better understand the applications of convolution. In the first figure, adjacent kernel is being used to blur the original image while in the second figure kernel is being used to identify all the edges in the image.

![Taj_Blur](http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/Gimp-Blur.png =400x200)

![Taj_edges](http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/Gimp-Edge.png =400x200)
___
2] 1x1 Convolution:
----------------
![1x1_convolution](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif =300x300)
<br/>
Whenever we see a neural network architecture with **1x1 kernel** size in convolutional layer, there are few questions comes into our mind, why this 1x1 convolutional is being used? Isn't it redundent?  

In simple terms, **1x1 convolutional** is being used for **dimensionality reduction** perpose i.e. *32* filters of *1x1* kernel size in convolution with an image of size *128x128* with *256* feature maps would result in size of *128x128x32*. So, it is **not redundent**. *1x1* convolution is also termed as **'feature pooling'** or **'cross channel parametric pooling'** technique since it does the sum pooling of features across all the feature-maps /channels for a given layer. This, 1x1 convolution, is a strictly linear transformation in filter space[[1]()] but in most applications, it is followed by non-linear activation layer i.e. **ReLU**, **Sigmoid**, **Tanh** etc. And yes, because of its small kernel size (1x1), it is **less suffered** by over-fitting. 

1x1 convolution was first introduced by the paper with title **[Network in Network](https://arxiv.org/pdf/1312.4400v3.pdf)**. According to paper "This cascaded cross channel parameteric pooling structure allows complex and learnable interactions of cross channel information".

In **[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)** architecture, 1x1 convolution in inception module, served several purposes:
1) Used as **dimension reduction** layer.
2) ReLU immediately after the 1x1 convolution adds more **non-linearity**.

![1x1_convolution_in_GoogLeNet](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/inception_1x1.png)

As shown in the image (b) Inception module with dimension reductions, *3x3* convolution and *5x5* convolution is preceded by *1x1* convolution and *3x3* max-pooling is followed by *1x1* convolution to reduce the dimensions with very little loss of information.
*1x1* convolution or cross channel information learning is biologically inspired from **human visual cortex containing receptive fields (kernels)**.
___
3] 10 examples of application of Mathjax:
-
1) **Convolution on two function**
    $$(f*g)(c) = \sum_{a} f(a).g(c-a)$$
    $$or$$
    $$(f*g)(c) = \int_{-\infty}^{\infty}f(a).g(c-a)dc$$
2) **Pythagorean Theorem**
    if $x$ and $y$ two vectors such that $x\bot y$ :
    $$ \lVert x+y \rVert^2  = \lVert x \rVert^2 + \lVert y \rVert^2$$

3) **Quadratic formula**
    When $a≠0$, there are two solutions to $ax^2+bx+c=0$ and they are
    $$x = {-b \pm \sqrt{b^2-4ac} \over 2a}$$
    
4) **Square expansion** 
   $$(a+b)^2 = a^2 + b^2+ 2ab$$

5) **Difference of squares**
   $$(a+b)(a-b) = a^2-b^2$$
   
6) **Logarithm**
   $y = \log_{b}(x)$ if and only if $x = b^y$ and $b\neq0$.
   
7) **Dot product**
   $$A.B= |A||B|cos\theta$$

8) **Sum of n natural numbers**
   $$S_n = \sum n(n+1)/2$$

9) **Sum of square of n natural numbers**
   $$S_n^2 = \sum n(n+1)(2n+1)/6$$

10) **Sum of cube of n natural numbers**
    $$S_n^3 = (\sum n(n+1)/2)^2$$
___
References:
-----------
- [Understanding Convolutions - colah's blog](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)
- [One by One [ 1 x 1 ] Convolution - counter-intuitively useful – Aaditya Prakash (Adi) – Random musings of a deep learning grad student](http://iamaaditya.github.io/2016/03/one-by-one-convolution/)
- [Convolution - Wikipedia](https://en.wikipedia.org/wiki/Convolution)
- [[1409.4842] Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [[1312.4400] Network In Network](https://arxiv.org/abs/1312.4400)
