# Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded1 = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded1 = Dense(784, activation='sigmoid')(encoded1)
# model
autoencoder = Model(input=input_img, output=decoded1)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')





# Regularized Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# add weight decay to parameters
encoded = Dense(encoding_dim, activation='relu'
	,kernel_regularizer=regularizers.l2(0.01))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
regularized_autoencoder = Model(input=input_img, output=decoded)
regularized_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Sparse Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# add sparse constraint to activation code
encoded = Dense(encoding_dim, activation='relu'
	,activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
sparse_autoencoder = Model(input=input_img, output=decoded)
sparse_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Denoising Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# corrupted input
corrupted_input = GaussianNoise(0.2)(input_img)
encoded = Dense(encoding_dim, activation='relu')(corrupted_input)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
denoising_autoencoder = Model(input=input_img, output=decoded)
denoising_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')