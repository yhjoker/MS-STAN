import torch
import torch.nn as nn
import torch.nn.functional as F
from STAC import STAC


class ResBlock(nn.Module): 
    def __init__( self, n_feats, kernel_size, bn=False, bias=True, activation = 'leakyrelu', res_scale=0.2):
        super(ResBlock, self).__init__()

        self.activation = activation

        if self.activation == 'leakyrelu':
            self.act = nn.LeakyReLU( negative_slope=0.2, inplace=True )
        elif self.activation == 'prelu':
            self.act = nn.PReLU()

        m = []
        for i in range(2):
            m.append( nn.Conv2d( n_feats, n_feats, kernel_size, padding=(kernel_size//2) , bias=bias) )
            if bn:
                m.append( nn.BatchNorm2d(n_feats) )
            if i == 0:
                m.append( self.act )

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResChain( nn.Module ):
    def __init__( self , input_step, n_resblocks , output_channels , kernel = 3 ):
        super( ResChain , self).__init__()
        self.input_step = input_step

        m = []
        for _ in range( n_resblocks ):
            m.append( ResBlock( n_feats = output_channels , kernel_size = kernel ) )
        
        self.res_chain = nn.Sequential( *m )

    def forward( self, x ):
        temp = []
        for i in range( self.input_step ):
            data = x[i]

            data = self.res_chain( data )
            temp.append( data )
        
        return torch.stack( temp )

class TResChain( nn.Module ):
    def __init__( self , input_step, n_resblocks , output_channels , kernel = 3 ):
        super( TResChain , self).__init__()
        self.input_step = input_step

        m = []
        for _ in range( n_resblocks ):
            m.append( ResBlock( n_feats = output_channels , kernel_size = kernel ) )
        
        self.res_chain = nn.Sequential( *m )

    def forward( self, x ):
        temp = []
        for i in range( self.input_step ):
            data = x[i]

            data = self.res_chain( data )
            temp.append( data )
        
        return torch.stack( temp )

class Tconv2d( nn.Module ):
    def __init__( self, input_step, input_channel, output_channel, kernel=3, bias=True ):
        super( Tconv2d , self).__init__()
        self.input_step = input_step

        self.op_conv = nn.Conv2d( input_channel, output_channel, kernel_size=kernel, padding=( kernel//2 ), bias=bias )

    def forward( self, x ):
        temp = []
        for i in range( self.input_step ):
            data = x[i]

            data = self.op_conv( data )
            temp.append( data )
        
        return torch.stack( temp )


class ResStackedBirectionalModule( nn.Module ):
    def __init__( self, i_step, n_resblocks, n_feats,
                hidden_channels1, kernel_size1, 
                hidden_channels2, kernel_size2, useGPU=True):
        super( ResStackedBirectionalModule, self ).__init__()
        self.step = i_step
        self.n_resblocks = n_resblocks
        self.useGPU = useGPU

        self.reschain_forward = TResChain( input_step=self.step, n_resblocks=self.n_resblocks, output_channels=n_feats )

        self.net_forward = STAC( n_feats, hidden_channels1, kernel_size1, step = i_step, effective_step= list( range( i_step ) ), useGPU = self.useGPU )

        self.input_conv = Tconv2d( input_step=self.step, input_channel=n_feats, output_channel=n_feats, kernel = 3 )
        
        self.net_backward = STAC( n_feats, hidden_channels2, kernel_size2, step = i_step, effective_step= list( range( i_step ) ), useGPU= self.useGPU )
 
        self.spatial_conv = Tconv2d( input_step=self.step, input_channel=n_feats, output_channel=n_feats, kernel = 3 )

        self.fusion_conv = Tconv2d( input_step=self.step, input_channel=n_feats * 2, output_channel=n_feats, kernel = 1 )

    def init_hidden( self, rank, batch_size, Height, Width ):
        net_forward_state = []
        net_backward_state = []

        for i in range( self.net_forward.num_layers ):    #为 forward encoder
            name = 'cell{}'.format(i)
            [ h, c ] = getattr( self.net_forward , name ).init_hidden( rank=rank, batch_size=batch_size, hidden= self.net_forward.hidden_channels[i], shape=( Height, Width ))
            net_forward_state.append( [ h, c ] )

        for i in range( self.net_backward.num_layers ):    #为 forward decoder
            name = 'cell{}'.format(i)
            [ h, c ] = getattr( self.net_backward , name ).init_hidden( rank= rank, batch_size=batch_size, hidden= self.net_backward.hidden_channels[i], shape=( Height , Width ))
            net_backward_state.append( [ h, c ] )
        
        return net_forward_state, net_backward_state

    def forward( self, x ):
        T, B, C, H, W = x.shape 
        rank = x.get_device()

        net_forward_state, net_backward_state = self.init_hidden( rank, B, H, W )

        x = self.reschain_forward( x )
        x = self.input_conv( x )

        forward_feat = x
        forward_output , net_forward_state = self.net_forward( forward_feat, net_forward_state )  # forward outputs, ( x, new_c )
        
        forward_output = torch.stack( forward_output )       

        backward_feat = torch.flip( forward_output, dims=[0] )
        
        backward_output , net_backward_state = self.net_backward( backward_feat , net_backward_state )

        backward_output = torch.stack( backward_output )
        backward_output = torch.flip( backward_output, dims = [0] )

        biLSTM_feat = self.fusion_conv( torch.cat( [ forward_output, backward_output ] ,dim = 2 ) )

        spatial_feat = self.spatial_conv( forward_feat )

        fusion_feat = biLSTM_feat + spatial_feat
        
        return fusion_feat

class TGlobalFusion( nn.Module ):
    def __init__( self, i_step, n_stage, n_feats, kernel_size ):
        super( TGlobalFusion, self ).__init__()
        self.step = i_step
        self.n_stage = n_stage
        self.fusion_conv = Tconv2d( input_step=self.step, input_channel=n_feats * n_stage, output_channel=n_feats, kernel = 1 )

    def forward( self, x ):
        x = self.fusion_conv( x )
        return x

class Reconstruction( nn.Module ):
    def __init__( self, i_step, n_feats, output_channel ):
        super( Reconstruction, self ).__init__()
        self.step = i_step
        
        self.n_resblocks = 8
        self.reschain = TResChain( input_step=self.step, n_resblocks=self.n_resblocks, output_channels=n_feats )

        self.upsample = nn.Sequential(  nn.Conv2d( n_feats, n_feats * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU( negative_slope=0.2, inplace=True ),
                                       nn.Conv2d( n_feats, n_feats * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU( negative_slope=0.2, inplace=True ), 
                                       nn.Conv2d( n_feats, n_feats, 3, 1, 1), nn.LeakyReLU( negative_slope=0.2, inplace=True ),
                                       nn.Conv2d( n_feats, output_channel, 3, 1, 1))
    def forward( self, x):
        x = self.reschain( x )
        temp = []
        for i in range( self.step ):
            data = x[i]

            data = self.upsample( data )
            temp.append( data )
        
        return torch.stack( temp )

class MSSTAN( nn.Module ):
    def __init__( self ,  upscale_factor , image_channel,  n_feats , n_resblocks, n_stage, kernel_size,i_step,
                hidden_channels1, kernel_size1, hidden_channels2, kernel_size2 ):
        super( MSSTAN , self ).__init__()

        self.step = i_step
        self.center = i_step // 2 
        self.upscale_factor = upscale_factor
        self.n_resblocks = n_resblocks
        self.n_stage = n_stage
        self.n_feats = n_feats
        self.kernel_size = kernel_size

        #extract features from input
        self.head = Tconv2d( input_step=self.step, input_channel=image_channel, output_channel=n_feats, kernel = 3 )

        self.net_stage = nn.ModuleList()
        for _ in range( n_stage ):
            self.net_stage.append( ResStackedBirectionalModule( i_step, n_resblocks, n_feats, hidden_channels1, kernel_size1, hidden_channels2, kernel_size2 ) )

        self.GFF = TGlobalFusion( self.step, self.n_stage, self.n_feats, self.kernel_size )

        self.construct = Reconstruction( self.step, n_feats, image_channel )

    def forward( self , x ):
        x = self.head( x )     
        stage_out = []
        for i in range( self.n_stage ):
            x = self.net_stage[i]( x )
            stage_out.append( x )        
        
        x = torch.cat( stage_out, dim=2 )

        x = self.GFF( x )
        x = self.construct( x )

        return x