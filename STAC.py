import torch
import torch.nn as nn
from torch.autograd import Variable


class STAlayer( nn.Module ):
    def __init__( self, input_feat, state_feat ):
        super( STAlayer, self ).__init__()

        self.emb1 = nn.Conv2d( input_feat, state_feat, 3, 1, 1 )
        self.emb2 = nn.Conv2d( state_feat, state_feat, 3, 1, 1 )
        
        self.diff = nn.Conv2d( 2 * state_feat, state_feat, 3, 1, 1 )

        self.STA_scale0 = nn.Conv2d( state_feat, state_feat, 3, 1, 1 )   # ( in, out, kernel, stride, padding )
        self.STA_scale1 = nn.Conv2d( state_feat, state_feat, 3, 1, 1 )
        self.STA_shift0 = nn.Conv2d( state_feat, state_feat, 3, 1, 1 )
        self.STA_shift1 = nn.Conv2d( state_feat, state_feat, 3, 1, 1 )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward( self, input, state ):
        emb1 = self.emb1( input )
        emb2 = self.emb2( state )

        feat = self.diff( torch.cat( [ emb1, emb2 ], dim=1 ) )
        scale = self.STA_scale1( self.lrelu( self.STA_scale0( feat ) ) )
        add = self.STA_shift1( self.lrelu( self.STA_shift0( feat ) ) )

        state = state * scale + add
        return state

class STACCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True , useGPU = False ):
        super(STACCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.useGPU = useGPU

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        #forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        #cell state
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        #output state
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None     
        self.STA_H = STAlayer( input_channels, hidden_channels )
        self.STA_C = STAlayer( input_channels, hidden_channels )


    def forward(self, x, h, c):
        h = self.STA_H( x, h )
        c = self.STA_C( x, c )

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, rank, batch_size, hidden, shape):
        if self.useGPU:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda( rank )
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda( rank )
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda( rank )
            return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda( rank ),  
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda( rank ))

class STAC(nn.Module):
    # input_channels corresponds to the first input feature map,only channels num is required
    # hidden_channels should be a list
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True , useGPU=False ):
        super(STAC, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)  #???
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self.useGPU = useGPU
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = STACCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i], self.bias , self.useGPU )
            setattr(self, name, cell)   
            self._all_layers.append(cell)

    def forward(self, input , internal_state ):
        #internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[ step ]                
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                # do forward
                [ h, c ] = internal_state[i]

                x, new_c = getattr(self, name)(x, h, c)     
                internal_state[i] = [ x, new_c ]

            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)       

        return outputs , internal_state