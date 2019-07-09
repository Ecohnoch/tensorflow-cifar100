from pretrained.cifar100 import cifar100

model = cifar100(model='resnet18')
model.test()

# model.test('....test')
# model.train('...train)