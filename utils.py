import torch
import torch.nn as nn
import os,argparse

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss

def save_model( model , optimizer, epoch, root_dir = 'model', index=0, n=0, hasGPU=True):        #保存特定训练 epoch 的模型
    if hasGPU :
        model_out_path = root_dir + '/'+"model_epoch_{}_gpu.pth".format( epoch )
    else:
        model_out_path = root_dir + '/'+"model_epoch_{}_cpu.pth".format( epoch )
    #check if dir exists
    if not os.path.exists( root_dir ):
        os.mkdir( root_dir )
    
    torch.save( 
        {
            'optimizer_dict' : optimizer.state_dict(),
            'model_dict' : model.state_dict() ,
            'loss_index' : index,
            'val_index':n
        }, model_out_path )
    print("model saved  ====> %s" % ( model_out_path ) )

def load_model( model , optimizer, epoch , root_dir = 'model' , hasGPU=True):        #加载特定训练 epoch 的模型
    if hasGPU:
        model_load_path = root_dir + '/' + "model_epoch_{}_gpu.pth".format( epoch )
    else:
        model_load_path = root_dir + '/' + "model_epoch_{}_cpu.pth".format( epoch )
        #model_load_path = 'model/' + "model_epoch_{}_gpu.pth".format( epoch )

    if hasGPU:
        checkpoint = torch.load( model_load_path )
        model.load_state_dict( checkpoint[ 'model_dict' ] )
        optimizer.load_state_dict( checkpoint[ 'optimizer_dict' ] )
        loss_index =  checkpoint[ 'loss_index' ]
        val_index =  checkpoint[ 'val_index' ]
    else:
        model.load_state_dict( torch.load( model_load_path , map_location='cpu') )
    
    print("model loaded ====> %s" %( model_load_path ) )
    return model, optimizer, loss_index, val_index


def adjust_learning_rate( optimizer , lr  ):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    cur_lr = lr * ( 0.1 )
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr