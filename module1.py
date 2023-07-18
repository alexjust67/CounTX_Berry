from functools import partial
import math
import copy
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
import numpy as np
from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed
import matplotlib.image as mpimg
import time
import cv2

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            size=(224,224),
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Feature interaction module (fim) specifics.
        self.fim_num_img_tokens = img_encoder_num_output_tokens

        # Use a fixed sin-cos embedding.
        self.fim_pos_embed = nn.Parameter(
            torch.zeros(1,self.fim_num_img_tokens, fim_embed_dim), requires_grad=False
        )

        self.fim_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    fim_embed_dim,
                    fim_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(fim_depth)
            ]
        )

        self.fim_norm = norm_layer(fim_embed_dim)

        # --------------------------------------------------------------------------
        # Density map decoder regresssion module specifics.
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(fim_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # CLIP model specifics (contains image and text encoder modules).
        self.clip_model = open_clip.create_model(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )

        # Freeze all the weights of the text encoder.
        vis_copy = copy.deepcopy(self.clip_model.visual)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.visual = vis_copy

    def initialize_weights(self):
        # Initialize the positional embedding for the feature interaction module.
        fim_pos_embed = get_2d_sincos_pos_embed(
            self.fim_pos_embed.shape[-1],
            int(self.fim_num_img_tokens**0.5),
            cls_token=False,
        )
        self.fim_pos_embed.data.copy_(
            torch.from_numpy(fim_pos_embed).float().unsqueeze(0)
        )

        # Initialize nn.Linear and nn.LayerNorm layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use Xavier uniform weight initialization following the official JAX ViT.
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_img_encoder(self, imgs):
        return self.clip_model.encode_image(imgs)

    def foward_txt_encoder(self, counting_queries, shot_num):
        # Exchange batch and shot dimensions.
        counting_queries = counting_queries.transpose(0, 1)  # [N, S, 77]->[S, N, 77]
        out_lst = []
        cnt = 0
        for counting_query_shot in counting_queries:
            cnt += 1
            if cnt > shot_num:
                break
            counting_query_tokens = self.clip_model.encode_text(counting_query_shot)
            N, C = counting_query_tokens.shape
            out_lst.append(counting_query_tokens)

        counting_query_tokens = torch.cat(out_lst, dim=0).reshape(shot_num, N, C)

        # Return batch and shot dimensions to their original order.
        counting_query_tokens = counting_query_tokens.transpose(
            0, 1
        )  # [S, N, C]->[N, S, C]

        return counting_query_tokens

    def forward_fim(self, img_tokens, txt_tokens):
        # Add positional embedding to image tokens.
        img_tokens = img_tokens + self.fim_pos_embed

        # Pass image tokens and counting query tokens through the feature interaction module.
        x = img_tokens
        for blk in self.fim_blocks:
            x = blk(x, txt_tokens)

        return self.fim_norm(x)

    def forward_decoder(self, fim_output_tokens):
        # Reshape the tokens output by the feature interaction module into a square feature map with [fim_embed_dim] channels.
        n, hw, c = fim_output_tokens.shape # 197 prob error
        h = w = int(math.sqrt(hw))
        x = fim_output_tokens.transpose(1, 2).reshape(n, c, h, w)

        # Upsample output of this map to be N x [fim_embed_dim] x 24 x 24, as it was in CounTR.
        x = F.interpolate(x, size=24, mode="bilinear", align_corners=False) # 14 ? 24

        # Pass [x] through the density map regression decoder and upsample output until density map is the size of the input image.
        x = F.interpolate(
            self.decode_head0(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head1(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head2(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head3(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the channel dimension from [x], as the density map only has 1 channel.
        return x.squeeze(-3)

    def forward(self, imgs, counting_queries, shot_num):
        img_tokens = self.forward_img_encoder(imgs)
        txt_tokens = self.foward_txt_encoder(counting_queries, shot_num)
        fim_output_tokens = self.forward_fim(img_tokens, txt_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred


def main_counting_network(**kwargs):
    model = CountingNetwork(
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def runmodel(image,text,model,device='cpu',showkern=False):

    # Preprocess the image.
    image = preprocess(image)
    #if showkern is true it will show the image.
    if showkern:
        plt.imshow(image.permute(1,2,0))
        plt.show()
    # Place the model in eval mode.
    model.eval()

    # The shot number is 1, since there is only 1 text description.
    shot_num = 1

    model = model.to(device)
    
    # Run the model.
    with torch.no_grad():
        density_map = model(image.unsqueeze(0).to(device), text.unsqueeze(0).to(device), shot_num)

    return density_map

def split_image(image,sq_size=224):
    
    #get image size and rescale it to the closest multiple of sq_size.
    w,h=image.size
    w2=math.floor(w/sq_size)*sq_size
    h2=math.floor(h/sq_size)*sq_size
    image=image.resize((w2,h2),Image.LANCZOS)
    image=np.array(image)
    first=True
    
    #split the image into sq_size x sq_size squares and return them as a numpy array of shape (n,224,224,3)
    for i in range(0, w2, sq_size):
        for j in range(0, h2, sq_size):
            if first:
                im=image[i:i+sq_size, j:j+sq_size]
                im=np.expand_dims(im, 3)
                first=False
            else:
                ima=np.expand_dims(image[j:j+sq_size, i:i+sq_size], 3)
                im=np.append(im,ima,axis=3)
    
    return w2,h2,np.transpose(im,(3,0,1,2))

def split_image_stride(image,sq_size=224,stride=50):#NB: stride here is intended as the overlap between squares.

    #get image size and find the closest multiple of sq_size rounded to the next integer to find the number of squares in each dimension, 
    #if autostride is enabled the stride will be set to not need to clip the last square to the bottom.
    w,h=image.size
    w2=math.ceil(w/sq_size)
    h2=math.ceil(h/sq_size)
    
    if stride[0]=='autostride':                                     #if autostride is enabled the stride will be set to not need to clip the last square to the bottom.
        stridex=round((math.ceil(w/sq_size)*sq_size-w)/w2)
        stridey=round((math.ceil(h/sq_size)*sq_size-h)/h2)
        if stridex<stride[1]: stridex=stride[1]
        if stridey<stride[1]: stridey=stride[1]
    else:
        stridex=stride[0]
        stridey=stride[1]

    image=np.array(image)
    first=True
    cor=[]
    inx=0
    iny=0
    for i in range(0, w, sq_size-stridex):
        for j in range(0, h, sq_size-stridey):
            
            #if it's the first square, create the array and append it.
            if first:
                im=image[i:i+sq_size, j:j+sq_size]
                im=np.expand_dims(im, 3)
                first=False

                #create a list of coordinates to map the squares to the original image as indexes [x,y] ex: [[0,0],[0,1],[0,2],[1,0],[1,1]] for a 3x2 image.
                cor.append([i,j])

            else:

                #if the square is too close to the bottom or right edge of the image, clip it to the edge.
                if j+sq_size>h: j=h-sq_size
                if i+sq_size>w: i=w-sq_size

                #append the square to the array.
                ima=np.expand_dims(image[j:j+sq_size, i:i+sq_size], 3)
                im=np.append(im,ima,axis=3)
                cor.append([inx,iny])
            iny+=1
        
        inx+=1
        iny=0


    return stridex,stridey,len(range(0, w, sq_size-stridex)),len(range(0, h, sq_size-stridey)),np.transpose(im,(3,0,1,2)),cor
    
def density_map_creator(image,model,text_add,query,dm_save,device="cpu",sqsz=224,stride=50,showkern=False,norm=0,shownorm=False):
    
    # Define preprocessing.
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    if stride==0:
        no_stride=True
    else:
        no_stride=False
    fir=True
    
    # Resize and center crop the image into sqsz shapes.
    if no_stride: 
        w1,h1,image2=split_image(copy.deepcopy(image),sqsz)
        deh=int((h1/sqsz)*384)  #actual size of the final image
        dew=int((w1/sqsz)*384)
        stridex=stridey=0
    else:
        dew,deh=image.size
        deh=math.floor((deh/sqsz)*384)  #actual size of the final image
        dew=math.floor((dew/sqsz)*384)
        stridex,stridey,w1,h1,image2,coord=split_image_stride(copy.deepcopy(image),sqsz,stride=stride)
    
    for text in query:

        enc_txt=tokenizer(text)
            
        if norm!= 0: image = normalize(image,norm,show=shownorm)

        w=0
        h=0
        density_map=torch.zeros((deh,dew))    #create a zero tensor with the dimension of the final image that will be no of squares * 384(standard model output)
        tot=0
        time1=[]
        time2=[]
        dens_time=time.time()

        for i in image2:                       #loop through the images
            
            if no_stride: 
                density_map[h:h+384,w:w+384]=runmodel(i,enc_txt,model,device=device,showkern=showkern).to(torch.device("cpu"))

            else:                               #if stride is used it will loop through the coordinates of the squares and run the model on them and then put the output in the right place in the final image using a max function between the overlapping squares
                
                inx=math.floor(coord[tot][0]*(384-(stridex/sqsz)*384))          #transform the coordinate indexes in the 384 coord space
                iny=math.floor(coord[tot][1]*(384-(stridey/sqsz)*384))
                if inx>dew-384: inx=dew-384
                if iny>deh-384: iny=deh-384
                denmap=runmodel(i,enc_txt,model,device=device,showkern=showkern).to(torch.device("cpu"))
                density_map[iny:iny+384,inx:inx+384]=torch.max(density_map[iny:iny+384,inx:inx+384],denmap)      

            if (h+384!=int((h1/sqsz)*384)and no_stride) or ((not no_stride) and h//384!=coord[len(coord)-1][1]):        #check if it has arrived to the bottom (only used by no_stride)
                
                h+=384
                
                time2.insert(0,time.time())                             #calculate the time remaining
                try:
                    time1.insert(0,time2[0]-time2[1])
                    time2.pop(2)
                    try:
                        time2.pop(3)
                    except:
                        pass
                except:
                    pass
                try:
                    avgperpoint=sum(time1[0:10])/len(time1[0:10])
                except:
                    avgperpoint=0
                
                if no_stride:                                            #print the progress
                    print(h//384," :  "+str(deh//384)+"    ",w//384," :  "+str(dew//384)+"    ",(tot)," : ",image2.shape[0],"  ",str(round((tot/image2.shape[0])*100,2))+"%"+ "    "+"Time remaining: ",round(avgperpoint*(image2.shape[0]-tot),0),"s"+"   ",text_add,"      ",end="\r")
                else:
                    print(h//384," : ",coord[len(coord)-1][1],"    ",w//384," : ",coord[len(coord)-1][0],"    ",(tot)," : ",image2.shape[0],"  ",str(round((tot/image2.shape[0])*100,2))+"%"+ "    "+"Time remaining: ",round(avgperpoint*(image2.shape[0]-tot),0),"s"+"   ",text_add,"      ",end="\r")

                tot+=1
            
            else:
                time2.insert(0,time.time())
                tot+=1
                h=0
                w=w+384
        
        print("Done calculating density map in:",round(time.time()-dens_time,2),text_add,"                                       ")

        if dm_save: np.save("./img/results/density_map.npy",density_map.numpy())        #save the density map if needed
        if fir:
            density_tot=density_map
            fir=False
        else:
            density_tot+=density_map
    
    density_map=density_tot/len(query)

    return density_map,dew,deh,stridex,stridey