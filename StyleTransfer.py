import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import imageio

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_model():
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.view(-1, 1, 1)
            self.std = std.view(-1, 1, 1)

        def forward(self, img):

            return (img - self.mean) / self.std
        
    vgg_model = models.vgg19(pretrained=True)
    for param in vgg_model.parameters():
        param.requires_grad_(False)
        
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    
    model = nn.Sequential()
    model.add_module("Normalization", Normalization(vgg_mean, vgg_std))
    i = 1
    j = 1
    for layer in vgg_model.features:
        if isinstance(layer, nn.Conv2d):
            model.add_module(f"Conv_{i}_{j}", layer)
            j += 1
        elif isinstance(layer, nn.ReLU):
            model.add_module(f"ReLU_{i}_{j}", layer)
        elif isinstance(layer, nn.MaxPool2d):
            model.add_module(f"Avgpool_{i}", nn.AvgPool2d(2, 2))
            j = 1
            i+= 1
    model = model.cuda().eval()
    return model

def load_images(content, style, imsize):

    tfms = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    
    content_img = Image.open(content)
    style_img = Image.open(style)
    
    
    # plt.imshow(content_img)
    # plt.title("Content")
    # plt.pause(0.001)
    # plt.imshow(style_img)
    # plt.title("Style")
    # plt.show()
    
    
    content_img = tfms(content_img).view(-1, 3, imsize, imsize).cuda()
    style_img = tfms(style_img).view(-1, 3, imsize, imsize).cuda()
    generated_img = content_img.clone().detach().requires_grad_(True)
    
    return content_img, style_img, generated_img

def gram_matrix(filters):
    b, c, h, w = filters.size()
    filters = filters.view(c, h * w)
    return torch.mm(filters, filters.t()).div(c*h*w)

def get_activation(x, model):
    
    layers = {1: 'Conv_1_1',
              6: 'Conv_2_1',
              11: 'Conv_3_1',
              20: 'Conv_4_1',
              22: 'Conv_4_2',
              29: 'Conv_5_1'}


    activations = {}
    for i, layer in enumerate(model.children()):
        x = layer(x)
        if i in layers:
            activations[layers[i]] = x
    return activations

def save_image(image, folder, i):
    image = image.clone().detach().cpu()
    image = image.squeeze(0)
    tfm = transforms.ToPILImage()
    image = tfm(image)
    imageio.imwrite(f"{folder}/{i}.jpg", image)    

content_img, style_img, generated_img = load_images( "neckarfront.jpg", "starry.jpg", 512)
criterion = torch.nn.MSELoss()
model = get_model()
opt = optim.Adam([generated_img], lr=0.01)
content_weight = 1
style_weight = 1e6
content_layer = "Conv_4_1"
style_activation = get_activation(style_img, model)
input_content = get_activation(content_img, model)[content_layer]
steps = 300

print("Started optimizing..")
for i in range(1, steps+1):
    opt.zero_grad()
    generated_img.data.clamp_(0, 1)
    output_activations = get_activation(generated_img, model)
    output_content = output_activations[content_layer]
    content_loss = criterion(input_content, output_content)
    style_loss = 0
    for key in style_activation:

        style_gram = gram_matrix(style_activation[key])
        output_gram = gram_matrix(output_activations[key])
        style_loss += criterion(style_gram, output_gram)
        
    loss = content_loss * content_weight + style_loss * style_weight
    loss.backward(retain_graph=True)
    opt.step()
    if i % 10 == 0:
        generated_img.data.clamp_(0, 1)
        print(f"[{i}/300], loss: {loss.item()}")
        save_image(generated_img, "default_progress", i)