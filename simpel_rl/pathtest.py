import os

model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model','critic')
print(model_dir)

if not os.path.exists(model_dir):
	os.makedirs(model_dir)

model_dir = os.path.join(model_dir, 'critic')
print('final directory are :', model_dir)