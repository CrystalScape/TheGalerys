import numpy as np 
from pathlib import Path
import PIL.Image as img 
import numpy as np 
import torch 
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import Softmax
from torch.nn import Flatten 
from torch.nn import BatchNorm2d
from torch.nn import LeakyReLU
from torch.nn import Dropout
from torch.nn import AvgPool2d
from torch.nn import Module
from torch.nn.functional import interpolate
import pandas as pd

class IMRDist(Module): 
    
    def __init__(self, num_chanel , num_prob,  *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.Conv = Conv2d(num_chanel , 16 , 3)
        self.BatchN = BatchNorm2d(3)
        self.Out = Linear(1 * 16 * 39 * 39 , num_prob)
        self.flats = Flatten()
        self.outScore = Softmax(1)
        self.Rel = LeakyReLU()
        self.Drops = Dropout(0.5)
        self.Avg = AvgPool2d(3)
        
    def forward(self , inputs): 
        x = self.BatchN(inputs)
        x = self.Conv(x)
        x = self.Avg(x)
        x = self.flats(x)
        x = self.Drops(x)
        x = self.Rel(x)
        x = self.Out(x)
        x = self.outScore(x)
        return x 
    
class MAB_Env : 
    
    def __init__(self , arms:int , image_dir:str) :
        self.arms = arms 
        self.images_link = image_dir
        self.devide = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def Build_state(self , arms): 
        bandits_matrix = torch.rand((arms , arms))
        return bandits_matrix
        
    def Update_position(self): 
        position = torch.randint(0 , self.arms-1 , (1,))[0]
        return position
    
    def get_position(self): 
        return self.Update_position()
    
    def get_probability(self , state):
        return state[self.get_position().detach().numpy()][self.arms-1]
    
    def image_Slection(self , turn_arte = 1.5 , seed = 123 , step = 0.01 , episode = 5): 
        torch.manual_seed(seed)
        list_dirs = Path(self.images_link)
        list_dirs = [item for item in list_dirs.iterdir()]
        list_dirs = [list_dirs[np.random.randint(0 , len(list_dirs)-1)] for _ in range(self.arms)]
        list_image_name = list_dirs[:self.arms]
        Scores = []
        img_names = []
        argmaxScore = []
        tau = 1.10
        # Build Softmax Exsploration
        softmax_exsploration = lambda x , tau : (
            np.exp(x.detach().numpy() / tau) / 
            np.sum(np.exp(x.detach().numpy() / tau)))
        # scoring
        for i in list_image_name :
            state = self.Build_state(self.arms)
            opens = img.open(i)
            print(i)
            vec = np.array(opens) / 255.0 
            vec = np.expand_dims(vec , 0)
            vec = np.transpose(vec , (0 , 3 , 1 , 2))
            vec = torch.from_numpy(vec)
            vec_new = interpolate(vec , (120 , 120) , mode='bilinear' , 
                                  align_corners=False).to(self.devide)
            vec_new = vec_new.to(torch.float32)
            Models = IMRDist(3 , self.arms)
            Optims = torch.optim.SGD(Models.parameters() , 0.001 , momentum=0.9)
            Optims.zero_grad()
            Models.train()
            Score = Models(vec_new)
            Score = Score * turn_arte
            Score = softmax_exsploration(Score , tau)
            Scores.append(Score[0][np.random.randint(0 , len(Score[0])-1)])
            img_names.append(i)
            argmaxScore.append(np.argmax(Score[0]))
            Optims.step()
            if torch.rand((1,))[0] < self.get_probability(state): 
                tau += step 
        values = {'ranks' : argmaxScore , 'image_dir' : img_names , 'probability' : Scores}
        values = pd.DataFrame(data = values).sort_values(by = 'ranks' , 
                                                         ascending=False).reset_index(drop=True).drop_duplicates()
        return values
        
             
Slec = MAB_Env(5 , '\mount\src\thegalerys\Selection\DataGambar\date')     
print(Slec.image_Slection())
      
            
        
    
