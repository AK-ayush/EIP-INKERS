# Assignment-2
---

Notebook: [code_9th.ipynb](./code_9th.ipynb)

Notebook with GAP: [code_9thgap.ipynb](./code_9th_gap.ipynb)

---

Training logs for first notebook:

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 11s 178us/step - loss: 0.5092 - acc: 0.8586 - val_loss: 0.0913 - val_acc: 0.9815
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 7s 110us/step - loss: 0.2532 - acc: 0.9259 - val_loss: 0.0657 - val_acc: 0.9839
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 7s 110us/step - loss: 0.1991 - acc: 0.9413 - val_loss: 0.0498 - val_acc: 0.9887
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 7s 109us/step - loss: 0.1700 - acc: 0.9458 - val_loss: 0.0416 - val_acc: 0.9901
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 7s 110us/step - loss: 0.1523 - acc: 0.9491 - val_loss: 0.0448 - val_acc: 0.9885
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 7s 110us/step - loss: 0.1387 - acc: 0.9524 - val_loss: 0.0311 - val_acc: 0.9915
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 7s 110us/step - loss: 0.1325 - acc: 0.9517 - val_loss: 0.0334 - val_acc: 0.9908
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 6s 108us/step - loss: 0.1245 - acc: 0.9542 - val_loss: 0.0267 - val_acc: 0.9927
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 7s 111us/step - loss: 0.1183 - acc: 0.9541 - val_loss: 0.0297 - val_acc: 0.9912
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 7s 113us/step - loss: 0.1138 - acc: 0.9544 - val_loss: 0.0269 - val_acc: 0.9921
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 7s 112us/step - loss: 0.1089 - acc: 0.9553 - val_loss: 0.0232 - val_acc: 0.9940
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 7s 112us/step - loss: 0.1080 - acc: 0.9542 - val_loss: 0.0239 - val_acc: 0.9927
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 7s 111us/step - loss: 0.1038 - acc: 0.9573 - val_loss: 0.0244 - val_acc: 0.9932
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 7s 110us/step - loss: 0.1025 - acc: 0.9565 - val_loss: 0.0237 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 106us/step - loss: 0.0990 - acc: 0.9573 - val_loss: 0.0252 - val_acc: 0.9935
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0970 - acc: 0.9588 - val_loss: 0.0220 - val_acc: 0.9941
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0981 - acc: 0.9567 - val_loss: 0.0248 - val_acc: 0.9933
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0967 - acc: 0.9567 - val_loss: 0.0218 - val_acc: 0.9945
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0945 - acc: 0.9575 - val_loss: 0.0224 - val_acc: 0.9941
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0935 - acc: 0.9581 - val_loss: 0.0224 - val_acc: 0.9942
Out[25]:
<keras.callbacks.History at 0x7f31d2ad1978>
```

And result of `model.evalute()` for  first notebook:

```python
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

[0.02243425567825325, 0.9942]
```

---
---
---

Training logs for second notebook:

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 14s 235us/step - loss: 0.4039 - acc: 0.9393 - val_loss: 0.0865 - val_acc: 0.9839
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 7s 120us/step - loss: 0.1088 - acc: 0.9817 - val_loss: 0.0517 - val_acc: 0.9893
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 7s 124us/step - loss: 0.0775 - acc: 0.9844 - val_loss: 0.0439 - val_acc: 0.9899
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 7s 120us/step - loss: 0.0612 - acc: 0.9863 - val_loss: 0.0362 - val_acc: 0.9904
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 7s 122us/step - loss: 0.0537 - acc: 0.9877 - val_loss: 0.0284 - val_acc: 0.9928
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 7s 123us/step - loss: 0.0474 - acc: 0.9891 - val_loss: 0.0284 - val_acc: 0.9915
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 7s 120us/step - loss: 0.0428 - acc: 0.9901 - val_loss: 0.0294 - val_acc: 0.9914
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 7s 121us/step - loss: 0.0389 - acc: 0.9901 - val_loss: 0.0305 - val_acc: 0.9916
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 7s 120us/step - loss: 0.0364 - acc: 0.9910 - val_loss: 0.0257 - val_acc: 0.9925
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 7s 122us/step - loss: 0.0351 - acc: 0.9915 - val_loss: 0.0237 - val_acc: 0.9921
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 7s 123us/step - loss: 0.0338 - acc: 0.9913 - val_loss: 0.0268 - val_acc: 0.9918
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 7s 120us/step - loss: 0.0329 - acc: 0.9912 - val_loss: 0.0238 - val_acc: 0.9933
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 7s 119us/step - loss: 0.0309 - acc: 0.9917 - val_loss: 0.0203 - val_acc: 0.9936
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 7s 119us/step - loss: 0.0305 - acc: 0.9918 - val_loss: 0.0217 - val_acc: 0.9940
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 7s 121us/step - loss: 0.0270 - acc: 0.9928 - val_loss: 0.0205 - val_acc: 0.9942
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 7s 122us/step - loss: 0.0269 - acc: 0.9930 - val_loss: 0.0197 - val_acc: 0.9936
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 7s 121us/step - loss: 0.0269 - acc: 0.9931 - val_loss: 0.0195 - val_acc: 0.9947
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 7s 120us/step - loss: 0.0252 - acc: 0.9929 - val_loss: 0.0186 - val_acc: 0.9947
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 7s 118us/step - loss: 0.0253 - acc: 0.9931 - val_loss: 0.0179 - val_acc: 0.9941
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 7s 122us/step - loss: 0.0244 - acc: 0.9932 - val_loss: 0.0184 - val_acc: 0.9943
Out[53]:
<keras.callbacks.History at 0x7f31c35ec390>
```

And the output of `model.evalute()` from the second notebook:

```python
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.0184028473936487, 0.9943]

```

### Strategy taken for the notebooks:

Both of these notebooks are based on provided notebook [EIGHTH.ipynb](https://colab.research.google.com/drive/1W56m2kaidSNlRnzGNPHqdzJn2nRTmmI_).  

I have added the `bias=False` in each instance of `Convolutional2D()` API so that there will be no baises used in any of the model architectures. 
  
Then, in the first notebook, I have reduced the number of kernels in first and second convolutional layer to bring down the total number of model parameters below to `14,937` while achieving the best accuracy of `99.45%` within `20` epochs.

And, in the second notebook, I have applied the `global Average Pooling` after applying the `1x1` convolutional to bring down the number of channels equal to number of classes. Total number of parameters used in this model are `14,072` while achieving the best accuracy of `99.47%` within `20` epochs.       

Please refer the notebook for deeper understanding.
