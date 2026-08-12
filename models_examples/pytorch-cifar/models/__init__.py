"""Definiciones de modelos para CIFAR-10.

Adaptado de kuangliu/pytorch-cifar (licencia MIT; véase LICENSE-THIRD-PARTY).
Se conservan únicamente las seis arquitecturas empleadas en el banco de
pruebas, que exponen las siete variantes evaluadas: VGG19, ResNet18,
ResNet50, MobileNetV2, DenseNet121, EfficientNetB0 y ViT.
"""

from .vgg import *
from .resnet import *
from .mobilenetv2 import *
from .densenet import *
from .efficientnet import *
from .vit import *
