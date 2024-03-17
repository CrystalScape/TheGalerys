from torch.nn import Linear
from torch.nn import Embedding
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch import bmm , device , cuda
from torch.nn.functional import softmax
from torch.nn import GELU
from torch.nn import ReLU , Softmax
from torch.nn import Module , CrossEntropyLoss
from torch import math
from torch.multiprocessing import Process
from torch import cat
from torch import arange
from torch.nn import ModuleList
from torch import long , save , load
from torch import randn , randint , clamp
from torch import zeros , argmax

devices = device("cuda" if cuda.is_available() else "cpu")

class ScaleDotProduct(Module): 
    
    def Enggine(self , q , k , v): 
        dim_k = k.size(-1)
        att = bmm(q , k.transpose(1,2)) / math.sqrt(dim_k)
        att_soft = softmax(att , dim = -1)
        out = bmm(att_soft , v)
        return out
    def __init__(self , ins , outs):
        super().__init__()
        self.q = Linear(ins , outs)
        self.k = Linear(ins , outs)
        self.v = Linear(ins , outs)
    def forward(self,inputs):
        x = self.Enggine(self.q(inputs) , self.k(inputs) , self.v(inputs))
        return x
    
class Attention(Module): 
    
    def __init__(self , emben_dim , num_head):
        super().__init__()
        head_dim = emben_dim // num_head
        self.out = Linear(emben_dim ,  emben_dim)
        self.head = ModuleList([ScaleDotProduct(emben_dim , head_dim) 
                                for _ in range(num_head)])
    def forward(self,inputs):
        x = cat([h(inputs) for h in self.head] , dim=-1)
        x = self.out(x)
        return x 
    
class EncoderLayers(Module): 
    
    def __init__(self , hiddens , embendim , num_head):
        super().__init__()
        self.LNorms1 = LayerNorm(hiddens)
        self.LNorms2 = LayerNorm(hiddens)
        self.Attentionss = Attention(embendim , num_head)
    def forward(self, inputs): 
        x = inputs + self.Attentionss(self.LNorms1(inputs))
        return x 
    
class Positional(Module):
    
    def __init__(self,Vocab_size ,Hiddens_emb) -> None:
        super().__init__()
        self.Embens = Embedding(Vocab_size , Hiddens_emb)
        self.Lnorms = LayerNorm(Hiddens_emb , eps=1e-12)
    def forward(self , inputs):
        inputs = inputs.to(devices).long()
        inputs = clamp(inputs , 0 , self.Embens.num_embeddings-1)
        token_id = self.Embens(inputs)
        embending = token_id  
        embending = self.Lnorms(embending)
        return embending 
    
class GeneticMutationLayer1D(Module):
    r'''## Genetic Mutation Layer
    adalah sebuah Layer Neural Network yang terinspirasi dari sebuah
    algoritma genetic (Genetic Algorithm). 
    
    yang di mana terdapat tahapan pemrosesan
    1. Membuat populasi 
    >>> pop = randn / popRate
    
    2. Melakukan mutasi
    >>> pos = randint(0,size[0])
    >>> child1 = zeros(size) ; child2 = zeros(size)
    >>> child[0:pos] = gen1[0:pos] ; child1[:pos] = gen2[:pos] # sama untuk child 2 nya
    >>> out = child1 + childd2 * rate 
    >>> return out'''
    def __init__(self, units_in , units_out , 
                 populationrate = 1.5 , MutateRate = 0.01 , relu = False):
        super().__init__()
        self.Lins = Linear(units_in, units_out)
        self.rels = ReLU()
        self.PopRate = populationrate
        self.mRate = MutateRate
        self.rel = relu
    def SpawnPopulation(self , size , Population_rate = 1.5):
        return randn(size)/Population_rate
    def Mutate(self,gen1 , gen2 , rate = 0.01): 
        shape = gen1.shape[0]
        child1 = zeros(gen1.shape)
        child2 = zeros(gen1.shape)
        pos = randint(0,shape,(1,))[0]
        child1[0:pos] = gen1[0:pos]
        child1[pos:] = gen2[pos:]
        child2[0:pos] = gen1[0:pos]
        child2[pos:] = gen2[pos:]
        out = (child1 + child2)*rate
        return out
    def forward(self,inputs): 
        gen1 = self.SpawnPopulation(inputs.shape , self.PopRate)
        gen2 = inputs
        Mutation = self.Mutate(gen1 , gen2 , self.mRate)
        out = self.Lins(Mutation)
        if self.rel : 
            out = self.rels(out)
        return out
    
class Encoder(Module): 
    def __init__(self , vocab_size , hidden_layers , Droprate , 
                 embendim , N_head , num_layers):
        super().__init__()
        self.pos = Positional(vocab_size , hidden_layers)
        self.EncoLayers = ModuleList([EncoderLayers(hidden_layers ,
                                                   embendim , 
                                                   N_head) for _ in range(num_layers)])
    def forward(self , inputs): 
        x = self.pos(inputs)
        for l in self.EncoLayers: 
            x = l(x)
        return x 
    
class GTN(Module): 
    def __init__(self , n_label , vocab_size , hidden_layers , Droprate , 
                 embendim , N_head , num_layers):
        super().__init__()
        self.Encoder = Encoder(vocab_size , hidden_layers , 
                               Droprate , embendim , N_head , num_layers)
        self.Drop = Dropout(Droprate)
        self.Out = Linear(hidden_layers , n_label)
        self.Mutated = GeneticMutationLayer1D(hidden_layers , hidden_layers , relu=True)
        self.outsoft = Softmax(1)
    def forward(self , inputs): 
        x = self.Encoder(inputs)
        x = self.Mutated(x)[:, 0 ,:]
        x = self.Drop(x)
        x = self.Out(x)
        x = self.outsoft(x)
        return x 
        
        