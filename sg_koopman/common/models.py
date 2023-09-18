'''
This modules defines neural network models for different algorithms.
'''
import torch
import sg_koopman.parameters as param


class FdynBrNet(torch.nn.Module):
    """
    This class defines the NN for the follower's feedback dybamics with BR.
    """
    def __init__(self) -> None:
        #torch.set_default_dtype(torch.float64)
        super().__init__()
        self.l1_size = param.nn_fdyn['linear1']
        self.l2_size = param.nn_fdyn['linear2']
        self.l3_size = param.nn_fdyn['linear3']
        self.dimin = self.l1_size[0]
        self.dimout = self.l3_size[1]

        self.linear1 = torch.nn.Linear(self.l1_size[0], self.l1_size[1]) # 8, 25
        self.linear2 = torch.nn.Linear(self.l2_size[0], self.l2_size[1]) # 25, 25
        self.linear3 = torch.nn.Linear(self.l3_size[0], self.l3_size[1]) # 25, 3
        self.activation = torch.nn.ReLU()

        # random initialization
        torch.manual_seed(param.seed)
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)


    def forward(self, xf, xl, ul):
        if xf.ndim > 1:
            y = torch.cat((xf, xl, ul), dim=1)
        else:
            y = torch.cat((xf, xl, ul), dim=0)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y
    

    def get_input_jac(self, xf, xl, u):
        """
        This function computes the jacobian of brnet w.r.t. input x and u.
        """
        # register hook for inner layer outpuut
        y = []  # y[i] is a 2d array
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        h2 = self.linear2.register_forward_hook(forward_hook)
        h3 = self.linear3.register_forward_hook(forward_hook)
        _ = self.forward(xf, xl, u)
        h1.remove()
        h2.remove()
        h3.remove()
        
        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is n x n matrix.
            """
            #df = torch.diag(1 - torch.tanh(y)**2)  # for tanh(x)
            df = torch.diag(1. * (y > 0))           # for relu(x)
            return df
        p = self.state_dict()
        jac = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight']
        jac_xf = jac[:, : param.dimxf]
        jac_xl = jac[:, param.dimxf: param.dimxf+param.dimxl]
        jac_ul = jac[:, param.dimxf+param.dimxl: ]
        #jac_x = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : param.dimxf]
        #jac_u = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, param.dimxf: ]
        return jac_xf, jac_xl, jac_ul



class KpNetPartial(torch.nn.Module):
    """
    This class defines the koopman operator bases for estimating the follower's dynamics only.
    """
    def __init__(self) -> None:
        #torch.set_default_dtype(torch.float64)
        super().__init__()
        self.l1_size = param.kp_partial['linear1']
        self.l2_size = param.kp_partial['linear2']
        self.l3_size = param.kp_partial['linear3']
        self.dimin = self.l1_size[0]    #param.dimxf
        self.dimout = self.l3_size[1]
        
        self.linear1 = torch.nn.Linear(self.l1_size[0], self.l1_size[1])
        self.linear2 = torch.nn.Linear(self.l2_size[0], self.l2_size[1])
        self.linear3 = torch.nn.Linear(self.l3_size[0], self.l3_size[1])
        self.activation = torch.nn.ReLU()

        # random initialization
        torch.manual_seed(param.seed)
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)

    
    def forward(self, xf):
        y = self.linear1(xf)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y
