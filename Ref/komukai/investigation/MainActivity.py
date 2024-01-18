#coding:Shift_JIS
"""
CNN-AE
FC-VAE
FC-CVAE
を同時に呼び出して、動かす

"""
import vae
import cnn
import cvae
import ae

epochs = 20
list1 = [1,2,4,8]
list2 = [1,2,4]

for i in list2:
	for j in list1:
		intermediate_dim = 392//i
		latent_dim = 40//j
		
		
		print("************************************************************************")
		print("AE:intermediate_dim="+str(intermediate_dim)+",latent_dim"+str(latent_dim))
		print("************************************************************************")
		ae.autoencoder(intermediate_dim = intermediate_dim,latent_dim = latent_dim,epochs = epochs)
		
		print("************************************************************************")
		print("VAE:intermediate_dim="+str(intermediate_dim)+",latent_dim"+str(latent_dim))
		print("************************************************************************")
		#vae.varietionnal_autoencoder(intermediate_dim = intermediate_dim,latent_dim = latent_dim,epochs = epochs)
		
		
		#print("************************************************************************")
		#print("CVAE:intermediate_dim="+str(intermediate_dim)+",latent_dim"+str(latent_dim))
		#print("************************************************************************")
		#cvae.conditionalvae(intermediate_dim=intermediate_dim,latent_dim = latent_dim,epochs = epochs)



	