import requests
import torch
from PIL import Image
from torchvision import transforms

from nets.blip import blip_itm

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

image_size = 384
image = load_demo_image(image_size=image_size,device=device)

model_url = 'model_data/model_base_retrieval_coco.pth'
    
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device='cuda')

caption = 'a woman sitting on the beach with a dog'
print('text: %s' %caption)

itm_output = model(image,caption,match_head='itm')
itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:, 1]
print('The image and text is matched with a probability of %.4f'%itm_score)

itc_score = model(image,caption,match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)