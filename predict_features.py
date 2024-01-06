import requests
import torch
from PIL import Image
from torchvision import transforms

from nets.blip import blip_feature_extractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size, device):
    img_path = 'img/demo.jpg' 
    raw_image = Image.open(img_path).convert('RGB')   
    
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
    ) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

image_size = 224
image = load_demo_image(image_size=image_size, device=device)     

model_url = 'model_data/model_base.pth'
    
model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

caption = 'a woman sitting on the beach with a dog'

multimodal_feature = model(image, caption, mode='multimodal')[0,0]
image_feature = model(image, caption, mode='image')[0,0]
text_feature = model(image, caption, mode='text')[0,0]
print("multimodal_feature: ", multimodal_feature.size())
print("image_feature: ", image_feature.size())
print("text_feature: ", text_feature.size())