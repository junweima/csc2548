# Text to Image generation

# March 10, 2018
	* created inverse gans

# March 19, 2018
	* added is_cuda check for train.py
	* using --split=2 for inference (testing set)
	* additional loss used by the original code author:
		g_loss = criterion(outputs, real_labels) \
						 + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
						 + self.l1_coef * l1_loss(fake_images, right_images)


# GAN losses
	* forward GAN: 
		* disc_loss = real_loss + fake_loss + wrong_loss(cls)(wrong image + right embedding)
		* gen_loss = g_loss
	* inverse GAN: 
		* disc_loss = real_loss + fake_loss + wrong_loss(cls)(wrong embedding + right image)
		* gen_loss = criterion(outputs, real_labels) # that's it?

# Questions:
	* cls: what would cls loss be for disc_loss in inverse GAN?
	


# TODO:
1. prepare dataset COCO (dataloader function)
2. generate word embeddings (using skip thought or gensim?)
3. GAN for image generation from word embeddings
4. GAN for caption generation
5. Cycle GAN structure







