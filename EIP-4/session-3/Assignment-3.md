# Assignment-3
---

Notebook: [code_9th.ipynb](./code_9th.ipynb)

---

Final Validation accuracy for Base Network:

```
Accuracy on test data is: 82.86
```

Updated model definition:

```python
model = Sequential()
model.add(SeparableConv2D(48, 3, padding="same", input_shape=(32, 32, 3))) # output: 32x32x48, RF: 3x3
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(48, 3))                                          # output: 30x30x48, RF: 5x5
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(SeparableConv2D(96, 3, padding="same"))                          # output: 30x30x96, RF: 7x7
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(96, 3))                                          # output: 28x28x96, RF: 9x9
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(SeparableConv2D(96, 3, padding="same"))                          # output: 28x28x96, RF: 11x11
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(192, 3))                                         # output: 26x26x192, RF: 13x13
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                                  # output: 13x13x192, RF: 14x14
model.add(Dropout(0.25))

model.add(SeparableConv2D(96, 3, padding="same"))                          # output: 13x13x96, RF: 18x18
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(96, 3))                                          # output: 11x11x96, RF: 22x22
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                                  # output: 5x5x96, RF: 24x24
model.add(Dropout(0.25))

model.add(SeparableConv2D(num_classes, 3))                                 # output: 3x3x10, RF: 32x32
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.1))
model.add(GlobalAveragePooling2D())                                        # output: 10, RF: 40x40

model.add(Activation('softmax'))                                           # output: 10, RF: 40x40

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Summary of the model:

```
Model: "sequential_30"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_199 (Separa (None, 32, 32, 48)        219       
_________________________________________________________________
batch_normalization_23 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_223 (Activation)  (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_200 (Separa (None, 30, 30, 48)        2784      
_________________________________________________________________
batch_normalization_24 (Batc (None, 30, 30, 48)        192       
_________________________________________________________________
activation_224 (Activation)  (None, 30, 30, 48)        0         
_________________________________________________________________
dropout_116 (Dropout)        (None, 30, 30, 48)        0         
_________________________________________________________________
separable_conv2d_201 (Separa (None, 30, 30, 96)        5136      
_________________________________________________________________
batch_normalization_25 (Batc (None, 30, 30, 96)        384       
_________________________________________________________________
activation_225 (Activation)  (None, 30, 30, 96)        0         
_________________________________________________________________
separable_conv2d_202 (Separa (None, 28, 28, 96)        10176     
_________________________________________________________________
batch_normalization_26 (Batc (None, 28, 28, 96)        384       
_________________________________________________________________
activation_226 (Activation)  (None, 28, 28, 96)        0         
_________________________________________________________________
dropout_117 (Dropout)        (None, 28, 28, 96)        0         
_________________________________________________________________
separable_conv2d_203 (Separa (None, 28, 28, 96)        10176     
_________________________________________________________________
batch_normalization_27 (Batc (None, 28, 28, 96)        384       
_________________________________________________________________
activation_227 (Activation)  (None, 28, 28, 96)        0         
_________________________________________________________________
separable_conv2d_204 (Separa (None, 26, 26, 192)       19488     
_________________________________________________________________
batch_normalization_28 (Batc (None, 26, 26, 192)       768       
_________________________________________________________________
activation_228 (Activation)  (None, 26, 26, 192)       0         
_________________________________________________________________
max_pooling2d_77 (MaxPooling (None, 13, 13, 192)       0         
_________________________________________________________________
dropout_118 (Dropout)        (None, 13, 13, 192)       0         
_________________________________________________________________
separable_conv2d_205 (Separa (None, 13, 13, 96)        20256     
_________________________________________________________________
batch_normalization_29 (Batc (None, 13, 13, 96)        384       
_________________________________________________________________
activation_229 (Activation)  (None, 13, 13, 96)        0         
_________________________________________________________________
separable_conv2d_206 (Separa (None, 11, 11, 96)        10176     
_________________________________________________________________
batch_normalization_30 (Batc (None, 11, 11, 96)        384       
_________________________________________________________________
activation_230 (Activation)  (None, 11, 11, 96)        0         
_________________________________________________________________
max_pooling2d_78 (MaxPooling (None, 5, 5, 96)          0         
_________________________________________________________________
dropout_119 (Dropout)        (None, 5, 5, 96)          0         
_________________________________________________________________
separable_conv2d_207 (Separa (None, 3, 3, 10)          1834      
_________________________________________________________________
batch_normalization_31 (Batc (None, 3, 3, 10)          40        
_________________________________________________________________
activation_231 (Activation)  (None, 3, 3, 10)          0         
_________________________________________________________________
dropout_120 (Dropout)        (None, 3, 3, 10)          0         
_________________________________________________________________
global_average_pooling2d_15  (None, 10)                0         
_________________________________________________________________
activation_232 (Activation)  (None, 10)                0         
=================================================================
Total params: 83,357
Trainable params: 81,801
Non-trainable params: 1,556
_________________________________________________________________
```

---

Training logs for 50 epochs:

```
Epoch 1/50
390/390 [==============================] - 40s 102ms/step - loss: 1.7056 - acc: 0.4199 - val_loss: 1.4925 - val_acc: 0.4953
Epoch 2/50
390/390 [==============================] - 34s 88ms/step - loss: 1.2667 - acc: 0.5920 - val_loss: 1.7913 - val_acc: 0.3957
Epoch 3/50
390/390 [==============================] - 34s 88ms/step - loss: 1.0809 - acc: 0.6510 - val_loss: 1.0718 - val_acc: 0.6413
Epoch 4/50
390/390 [==============================] - 34s 88ms/step - loss: 0.9573 - acc: 0.6926 - val_loss: 1.2440 - val_acc: 0.5643
Epoch 5/50
390/390 [==============================] - 34s 88ms/step - loss: 0.8686 - acc: 0.7187 - val_loss: 0.9020 - val_acc: 0.6978
Epoch 6/50
390/390 [==============================] - 34s 88ms/step - loss: 0.8066 - acc: 0.7383 - val_loss: 0.8131 - val_acc: 0.7204
Epoch 7/50
390/390 [==============================] - 34s 87ms/step - loss: 0.7502 - acc: 0.7565 - val_loss: 0.7846 - val_acc: 0.7425
Epoch 8/50
390/390 [==============================] - 34s 88ms/step - loss: 0.7087 - acc: 0.7684 - val_loss: 0.7738 - val_acc: 0.7460
Epoch 9/50
390/390 [==============================] - 34s 88ms/step - loss: 0.6748 - acc: 0.7800 - val_loss: 0.7902 - val_acc: 0.7362
Epoch 10/50
390/390 [==============================] - 34s 88ms/step - loss: 0.6452 - acc: 0.7888 - val_loss: 0.6837 - val_acc: 0.7715
Epoch 11/50
390/390 [==============================] - 35s 89ms/step - loss: 0.6167 - acc: 0.7978 - val_loss: 0.8572 - val_acc: 0.7129
Epoch 12/50
390/390 [==============================] - 34s 88ms/step - loss: 0.5950 - acc: 0.8040 - val_loss: 0.6231 - val_acc: 0.7957
Epoch 13/50
390/390 [==============================] - 34s 88ms/step - loss: 0.5738 - acc: 0.8118 - val_loss: 0.6937 - val_acc: 0.7659
Epoch 14/50
390/390 [==============================] - 34s 88ms/step - loss: 0.5577 - acc: 0.8165 - val_loss: 0.5947 - val_acc: 0.8001
Epoch 15/50
390/390 [==============================] - 34s 88ms/step - loss: 0.5434 - acc: 0.8187 - val_loss: 0.6192 - val_acc: 0.7920
Epoch 16/50
390/390 [==============================] - 34s 88ms/step - loss: 0.5257 - acc: 0.8241 - val_loss: 0.7780 - val_acc: 0.7460
Epoch 17/50
390/390 [==============================] - 35s 89ms/step - loss: 0.5103 - acc: 0.8312 - val_loss: 0.5869 - val_acc: 0.8080
Epoch 18/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4970 - acc: 0.8363 - val_loss: 0.6094 - val_acc: 0.8009
Epoch 19/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4883 - acc: 0.8361 - val_loss: 0.5586 - val_acc: 0.8125
Epoch 20/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4783 - acc: 0.8398 - val_loss: 0.6216 - val_acc: 0.7930
Epoch 21/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4662 - acc: 0.8449 - val_loss: 0.6606 - val_acc: 0.7768
Epoch 22/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4596 - acc: 0.8471 - val_loss: 0.5538 - val_acc: 0.8178
Epoch 23/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4448 - acc: 0.8523 - val_loss: 0.5221 - val_acc: 0.8250
Epoch 24/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4389 - acc: 0.8522 - val_loss: 0.5413 - val_acc: 0.8209
Epoch 25/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4353 - acc: 0.8541 - val_loss: 0.5534 - val_acc: 0.8134
Epoch 26/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4256 - acc: 0.8586 - val_loss: 0.5608 - val_acc: 0.8182
Epoch 27/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4188 - acc: 0.8595 - val_loss: 0.5287 - val_acc: 0.8215
Epoch 28/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4069 - acc: 0.8623 - val_loss: 0.5964 - val_acc: 0.8008
Epoch 29/50
390/390 [==============================] - 35s 90ms/step - loss: 0.4027 - acc: 0.8647 - val_loss: 0.5750 - val_acc: 0.8183
Epoch 30/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3980 - acc: 0.8675 - val_loss: 0.5583 - val_acc: 0.8140
Epoch 31/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3874 - acc: 0.8686 - val_loss: 0.5420 - val_acc: 0.8231
Epoch 32/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3883 - acc: 0.8693 - val_loss: 0.5142 - val_acc: 0.8331
Epoch 33/50
390/390 [==============================] - 35s 91ms/step - loss: 0.3777 - acc: 0.8732 - val_loss: 0.5192 - val_acc: 0.8339
Epoch 34/50
390/390 [==============================] - 36s 92ms/step - loss: 0.3711 - acc: 0.8739 - val_loss: 0.4921 - val_acc: 0.8381
Epoch 35/50
390/390 [==============================] - 36s 91ms/step - loss: 0.3703 - acc: 0.8757 - val_loss: 0.4998 - val_acc: 0.8346
Epoch 36/50
390/390 [==============================] - 36s 92ms/step - loss: 0.3654 - acc: 0.8772 - val_loss: 0.4997 - val_acc: 0.8408
Epoch 37/50
390/390 [==============================] - 36s 92ms/step - loss: 0.3603 - acc: 0.8780 - val_loss: 0.4982 - val_acc: 0.8355
Epoch 38/50
390/390 [==============================] - 36s 92ms/step - loss: 0.3556 - acc: 0.8797 - val_loss: 0.5566 - val_acc: 0.8214
Epoch 39/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3486 - acc: 0.8818 - val_loss: 0.5321 - val_acc: 0.8249
Epoch 40/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3486 - acc: 0.8823 - val_loss: 0.5124 - val_acc: 0.8353
Epoch 41/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3388 - acc: 0.8852 - val_loss: 0.5058 - val_acc: 0.8371
Epoch 42/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3411 - acc: 0.8834 - val_loss: 0.5033 - val_acc: 0.8398
Epoch 43/50
390/390 [==============================] - 35s 91ms/step - loss: 0.3319 - acc: 0.8860 - val_loss: 0.4660 - val_acc: 0.8465
Epoch 44/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3347 - acc: 0.8866 - val_loss: 0.4614 -  val_acc: 0.8536
Epoch 45/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3303 - acc: 0.8886 - val_loss: 0.5094 - val_acc: 0.8343
Epoch 46/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3295 - acc: 0.8880 - val_loss: 0.5174 - val_acc: 0.8365
Epoch 47/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3221 - acc: 0.8912 - val_loss: 0.4600 - val_acc: 0.8517
Epoch 48/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3106 - acc: 0.8951 - val_loss: 0.4867 - val_acc: 0.8400
Epoch 49/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3146 - acc: 0.8941 - val_loss: 0.5485 - val_acc: 0.8284
Epoch 50/50
390/390 [==============================] - 35s 90ms/step - loss: 0.3107 - acc: 0.8956 - val_loss: 0.4957 - val_acc: 0.8384
Model took 1752.82 seconds to train

Accuracy on test data is: 83.84
```



Best accuracy in 50 epochs is `85.36`.
