import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb

# 64 x 1024
class generator_inverse(nn.Module):
	def __init__(self):
		super(generator_inverse, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netG_I = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),

			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		# projection to 1024 sized embedding vector
		self.netG_I_projection = nn.Sequential(
			# state size. (ndf*2) x 4 x 4
			nn.Linear(self.ndf * 2 * 4 * 4, self.embed_dim)
		)

	def forward(self, inp):
		pdb.set_trace()
		x_intermediate = self.netG_I(inp)
		x_intermediate = x_intermediate.view(-1, self.ndf*2*4*4)
		reconstructed_embed = self.netG_I_projection(x_intermediate)
		return reconstructed_embed


class discriminator_inverse(nn.Module):
	def __init__(self):
		super(discriminator_inverse, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)

		return x.view(-1, 1).squeeze(1) , x_intermediate