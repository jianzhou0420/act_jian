from torchvision.transforms import Compose
import torch
import clip
from clip.model import ModifiedResNet
import numpy as np


class ModifiedResNetFeatures(ModifiedResNet):
    '''modified from clip.model.ModifiedResNet'''

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


def get_clip_encoders():
    '''
    (You can treat this function as a python file to understand its local and global variables)
    Fetch the encoder_image and encoder_text from the CLIP model.
    Note: varible clip is the clip package, varible clip_model is the clip model.
    '''
    clip_model, image_transforms = clip.load("RN50")  # TODO: Transform 有问题
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    visual_model = ModifiedResNetFeatures(layers, output_dim, heads)
    visual_model.load_state_dict(clip_model.visual.state_dict())
    visual_model = visual_model.to('cuda')
    # normalize = clip_transforms.transforms[-1]

    # modify transform

    image_transforms_jian = Compose([
        image_transforms.transforms[0],  # Resize
        image_transforms.transforms[1],  # CenterCrop
        image_transforms.transforms[4],  # Normalize # before normalnized, the image is in range [0,1]
    ])

    def encode_text(text):
        '''modified from clip.model.CLIP.encode_text
           Only deleted the matrix multiplication with the text projection matrix (because we don't need project, our dim is 512)
        '''
        if type(text) is str:
            text = clip.tokenize(text).to('cuda')
        elif type(text) is torch.Tensor:
            pass

        x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        # Jian: Extract the eot embedding, features of end of sentence after transformer(x) is a kind of summary of the sentence please refer the self-attention mechanism's definition
        return x

    def encode_image(x):
        '''assume you have a torch tensor image and range [0,1]'''

        x = image_transforms_jian(x).to('cuda')

        with torch.no_grad():

            x = visual_model(x)
        return x['res5']

    return encode_image, encode_text


# test #
def example():
    ''' for testing this file'''
    # Load the model
    device = "cuda"
    encoder_image, encoder_text = get_clip_encoders()

    # Preprocess the image

    image1 = np.load("/home/jian/git_all/git_manipulation/act_jian/z_others/image_example.npy")
    image1 = torch.tensor(image1).float()
    image1 = encoder_image(image1).to(device)
    print(image1.shape)

    string1 = "a photo of a dog"
    text1 = encoder_text(string1).to(device)
    text1 = encoder_text(string1)
    print(text1.shape)


if __name__ == "__main__":
    example()
