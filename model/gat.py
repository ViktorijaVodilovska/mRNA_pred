import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GATv2Conv, LayerNorm, GraphNorm
import wandb


class GAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, layers, hidden_channels, attention_heads, attention_dropouts, dropouts, graph_norm=False):
        super(GAT, self).__init__()
        
        self.layers = layers
        
        convs, norms, drops = [], [], []
        for i in range(layers):
            convs.append(GATv2Conv(
                in_channels=hidden_channels[i-1] if i!=0 else node_dim, 
                out_channels=hidden_channels[i], 
                heads=attention_heads[i], 
                dropout=attention_dropouts[i],
                edge_dim=edge_dim,
                return_attention_weights=True,
                # TODO: alternatives to try for concat: concat=True / is concatenated and projected (NN) back down to a size of 64 
                concat = False,
                
                # default params that i might need to play with:
                # negative_slope 
                # add_self_loops 
                # fill_value 
                # bias 
                # share_weights 
                ))

            if graph_norm == True:
                norms.append(GraphNorm(in_channels=hidden_channels[i]))
            else:
                norms.append(LayerNorm(in_channels=hidden_channels[i]))
                
            drops.append(Dropout(p=dropouts[i]))
            
        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)
        
        self.output_dim = hidden_channels[-1]
                

    def forward(self, x, edge_index):
        for i in range(self.layers):
            x = self._convs[i](x, edge_index)
            
            print(x.shape())
            x = self._norms[i](x)
            x = self._drops[i](x)
            
            # wandb.log({f'attention_{i}':attn})
        
        return x
    
  
            
    
    
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Parsing argument")
    # parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
    # parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer")
    # parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    # parser.add_argument("--step_size", type=int, default=100, help="Step size of StepLR")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Gamma of StepLR")
    # parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
    # parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
    # parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
    # parser.add_argument("--num_worker", type=int, default=8, help="Number of workers for data loader")
    # parser.add_argument("--out_dir", type=str, default="out", help="Name of the output directory")
    # parser.add_argument(
    #     "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
    # )
    # args = parser.parse_args()
    
    
    
