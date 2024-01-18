#coding:Shift_JIS

import cnn


epochs = 20
encode_dim = 32#4,8,16,32
latent_dim = 8
list = [16,8,6,4,2]
latent_dim_list = [6,4,2,1]

#cnn.cnn_ae(encode_dim = 32,latent_dim=6,epochs = epochs)
#cnn.cnn_ae(encode_dim = 32,latent_dim=4,epochs = epochs)
cnn.cnn_ae(encode_dim = 32,latent_dim=2,epochs = epochs)
#cnn.cnn_ae(encode_dim = 32,latent_dim=1,epochs = epochs)

#cnn.cnn_ae(encode_dim = 16,latent_dim=6,epochs = epochs)
#cnn.cnn_ae(encode_dim = 16,latent_dim=4,epochs = epochs)
#cnn.cnn_ae(encode_dim = 16,latent_dim=2,epochs = epochs)
#cnn.cnn_ae(encode_dim = 16,latent_dim=1,epochs = epochs)


#cnn.cnn_ae(encode_dim = 8,latent_dim=6,epochs = epochs)
#cnn.cnn_ae(encode_dim = 8,latent_dim=4,epochs = epochs)
#cnn.cnn_ae(encode_dim = 8,latent_dim=2,epochs = epochs)
#cnn.cnn_ae(encode_dim = 8,latent_dim=1,epochs = epochs)



#↓いる？(後回し)

#cnn.cnn_ae(encode_dim = 4,latent_dim=6,epochs = epochs)
#cnn.cnn_ae(encode_dim = 4,latent_dim=4,epochs = epochs)
#cnn.cnn_ae(encode_dim = 4,latent_dim=2,epochs = epochs)
#cnn.cnn_ae(encode_dim = 4,latent_dim=1,epochs = epochs)


#cnn.cnn_ae(encode_dim = 2,latent_dim=6,epochs = epochs)
#cnn.cnn_ae(encode_dim = 2,latent_dim=4,epochs = epochs)
#cnn.cnn_ae(encode_dim = 2,latent_dim=2,epochs = epochs)
#cnn.cnn_ae(encode_dim = 2,latent_dim=1,epochs = epochs)

