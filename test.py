import torch
import numpy as np
import os,argparse
from os.path import join
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from utils import save_model, load_model
from arch import MSSTAN

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo)
    iw = iw - (iw%modulo)
    img = img.crop( (0, 0, ih, iw) )
    return img

def get_in_range_for_image( output_seq ):
    output_seq = output_seq * 255.
    output_seq[ output_seq < 0 ] = 0
    output_seq[ output_seq > 255.] = 255.
    return output_seq.astype( np.uint8 )

def getTensor( img_list ):
    img_in_np = [ np.array( im ) for im in img_list ]   # ( H, W, C )
    im_set = np.stack( img_in_np )                      # ( T, H, W, C )
    im = np.transpose( im_set, [ 0 , 3 , 1 , 2 ] )      # ( T, C, H, W )

    im = im[ np.newaxis, :, :, :, : ]                   # 1,T,C,H,W

    im = torch.from_numpy( im ).float()
    im = torch.div( im , 255 )

    return im

def TensorToNumpy( input ):
    input = input.detach().cpu()
    output_seq = input
    output_seq = torch.squeeze(  output_seq )
    output_seq = np.array( output_seq )     #T,C,H,W

    output_seq = np.transpose( output_seq, [ 0, 2, 3, 1] )

    output_seq = get_in_range_for_image( output_seq )

    return output_seq

def process_data_single( model , img_input, bicubic, save_path , input_step, idx ):
    if hasGPU:
        model = model.cuda()
        img_input = img_input.cuda()
        bicubic = bicubic.cuda()  

    T , B , C , H , W = img_input.shape

    with torch.no_grad():
        net_output  = model( img_input )
        prediction = bicubic + net_output

    output_seq = prediction.detach().squeeze().float().cpu().clamp_( *(0,1) )
    output_seq = np.array( output_seq )
    output_seq = output_seq[ T//2, :, :, : ]
    output_seq = np.transpose( output_seq, [ 1, 2, 0 ] )
    
    output_seq = ( output_seq * 255.0 ).round()
    output_seq = output_seq.astype( np.uint8 )

    center_img = Image.fromarray( output_seq )

    save_image_path = "%s/%08d.png"%( save_path , idx )
    center_img.save( save_image_path )
    print("%s saved!" % save_image_path )


def GT_test_single( model, process_data_root , test_vid_root, input_step, scale ): 
    if not os.path.exists( process_data_root ):
        os.mkdir( process_data_root )
    
    hr_files = os.listdir( test_vid_root )      
                                                
    for name in hr_files:
        datapath = test_vid_root + name

        save_path = process_data_root + name
        if not os.path.exists( save_path ):
            os.mkdir( save_path )

        GT_file = sorted([ p for p in os.listdir( datapath ) if p.lower().startswith('0')])
        
        data_len = len( GT_file )
        gap = input_step // 2

        for idx in range( gap, data_len - gap ):
            start = idx - gap
            end = idx + gap + 1
    
            img_orig = [ modcrop( Image.open( join( datapath, '%08d.png'%( i ) )), scale ) for i in list( range( start, end ) ) ]

            img_orig = getTensor( img_orig )            #1,T,C,H,W
            img_input = DUF_downsample( img_orig )      #1,T,C,H,W

            img_input = torch.squeeze( img_input, dim=0 )
            bicubic = F.interpolate( img_input, scale_factor=4, mode='bicubic')
            
            img_input = torch.unsqueeze( img_input, dim=0 )
            bicubic = torch.unsqueeze( bicubic, dim=0 )

            img_input = torch.transpose( img_input, 0, 1 ) #T,B,C,H,W
            bicubic = torch.transpose( bicubic, 0, 1 )     #T,B,C,H,W

            process_data_single( model , img_input, bicubic, save_path , input_step, idx )


parser = argparse.ArgumentParser(description="optional args for MS-STAN")
parser.add_argument('-GTDir', type=str, default="/path/to/GT/" , help='GT directory') 
parser.add_argument('-SRDir', type=str, default="/path/to/test/" , help='SR directory') 
parser.add_argument('-upscale', type=int, default=4 , help='upscale factor') 
parser.add_argument('-input_channel', type=int, default=3 , help='input image channel') 
parser.add_argument('-input_frms', type=int, default=5, help='consecutive input frames')

opts = parser.parse_args()

if torch.cuda.is_available():
        hasGPU = True
else:
        hasGPU = False

#architecture args
n_feats = 64
n_resblocks = 2
n_stage = 10
kernel_size = 3
step_input = 5

#convLSTM args
hidden_channels1 = [ n_feats ]
kernel_size1 = [ 3 ]
hidden_channels2 = [ n_feats ]
kernel_size2 = [ 3 ]

#训练时初始化模型
model = MSSTAN( opts.upscale , opts.input_channel,  n_feats , n_resblocks, n_stage, kernel_size, opts.input_frms,
                hidden_channels1, kernel_size1, hidden_channels2, kernel_size2 )

optimizer = optim.Adam( model.parameters(), lr = 1e-4, betas=(0.9, 0.999), eps=1e-8 )

model, optimizer, loss_index, val_index  = load_model( model, optimizer, 60, root_dir = './' )

if hasGPU:
    model.cuda()

GT_test_single( model, opts.SRDir, opts.GTDir, opts.input_frms, scale=opts.upscale )
