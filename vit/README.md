# VIT:
	# What a constant latent vector of the size D.
	# Classification Head? :
		# implemented using an mlp
	# position embedding + patch embedding = positional information	

Everything has a bias, question is what kind.
The model can be implemented in different variations, based on what kind of a bias they have:
1) Pure ViT ==> the one that was originally implemented. 
   	* has implicit bias (Assumptions that EMERGE from ARCHITECTURE, not during learning)
	* more global, less local
2) Hybird ==> this is later optimization added to the model.
	* uses CNN(inductive bias) + Self_Attention(implicit bias).
	* so local + global


MODEL of Choice : Pure ViT (minimalist)
