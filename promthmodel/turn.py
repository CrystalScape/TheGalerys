from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset , DataLoader
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from promthmodel.GTNs import GTN
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler , Adam
from tqdm.auto import tqdm
from torch import Tensor , save , device , cuda , multiprocessing , load
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils import rnn
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True

dt = pd.read_csv('data.csv')
dt = dt.drop(columns=['Unnamed: 0'])
dt_l = dt['label']
l = []
for i in dt_l: 
    if i not in l : 
        l.append(i)
n_label = len(l)

lb = Tensor(LabelEncoder().fit_transform(dt_l))
print(lb)

lb_Globals = LabelEncoder()
lb_Globals.fit(dt_l)

print(dt['promth'][0])
tokenizer = get_tokenizer('basic_english')
coba = tokenizer(dt['promth'][0])
def dataiters(datas): 
    for t in datas: 
        yield tokenizer(t)
print(dataiters(dt['promth']))
vocab = build_vocab_from_iterator(dataiters(dt['promth']))
text_pips = lambda x : vocab(x)
print(coba)
print(text_pips(coba))
print(Tensor(text_pips(coba)))
print(n_label)
if 'ku' in vocab : print(True)

class AutoTrain(): 
    
    def __init__(self , link_data1 , link_data2 , n_labels = n_label , batchs = 64 , 
                 shuffle = True , vocab_fn = vocab) :
        super().__init__()
        self.TrainData = link_data1
        self.TestData = link_data2 
        self.n_label = n_labels
        self.batchsize = batchs
        self.shuffels = shuffle
        self.Tokenizers = get_tokenizer('basic_english' , language='en')
        self.textpip = lambda x : vocab_fn(self.Tokenizers(x))
        self.lb = LabelEncoder()
        self.vocabs = len(vocab_fn)
        self.Device = device("cuda" if cuda.is_available() else "cpu")
        
    def Turn_to_iterator(self,data): 
        for t in data : 
            yield self.Tokenizers(t)
            
    def LabelsEncoder(self , labels , Invers = False): 
        lab = self.lb.fit_transform(labels)
        if Invers : 
            lab = self.lb.inverse_transform(labels)
        return Tensor(lab)
    
    def DataText_To_Tensor(self): 
        Train = pd.read_csv(self.TrainData)
        Test = pd.read_csv(self.TestData)
        # Train 
        Train_result = []
        for tr in Train['promth']: 
            Train_result.append(Tensor(self.textpip(tr)))
        Train_result_x = rnn.pad_sequence(Train_result , batch_first=True , padding_value=0)
        Train_result_y = self.LabelsEncoder(Train['label'])
        Train_itertaor = TensorDataset(Train_result_x , Train_result_y)
        Train_Batch = DataLoader(Train_itertaor , batch_size=self.batchsize , 
                                 shuffle=self.shuffels)
        # Test 
        Test_result = []
        for ts in Test['promth']: 
            Test_result.append(Tensor(self.textpip(ts)))
        Test_result_x = rnn.pad_sequence(Test_result , True , 0)
        Test_result_y = self.LabelsEncoder(Test['label'])
        Test_iterstor = TensorDataset(Test_result_x , Test_result_y)
        Test_batch = DataLoader(Test_iterstor , batch_size=self.batchsize , 
                                shuffle=self.shuffels)
        return Train_Batch , Test_batch
    
    def Train(self , epoch = 150 , lr = 0.001):
        Models = GTN(self.n_label , self.vocabs , 768 , 
                     0.01 , 768 , 12 , 1).to(self.Device)
        Train_data , Test_data = self.DataText_To_Tensor()
        losses_fn = CrossEntropyLoss()
        Optims = Adam(params=Models.parameters() , lr=lr) 
        seduch = lr_scheduler.StepLR(Optims , lr)
        for i in tqdm(range(epoch)):
            Models.train() 
            total_acc = 0 
            total_cound = 0
            tt_acc = []
            tt_acc.append(total_acc)
            for indx , (t , l) in enumerate(Train_data) : 
                Optims.zero_grad()
                predict =  Models(t)
                loss = losses_fn(predict , l.long())
                loss.backward()
                Optims.step()
                total_acc += (predict.argmax(1) == l).sum().item()
                total_cound += l.size(0)
                if indx % 10 == 0 : 
                    print(f'Epoch Train : {i} , Loss : {loss:.3f} , Acc : {(total_acc / total_cound):.3f}')
            Models.eval() 
            total_acc_te = 0 
            total_cound_te = 0
            with torch.no_grad():
                for indxs , (te , le) in enumerate(Test_data): 
                    predicts = Models(te)
                    losses = losses_fn(predicts , le.long())
                    total_acc_te += (predicts.argmax(1) == le).sum().item()
                    total_cound_te += le.size(0)
                    if indxs % 10 == 0 : 
                        print(f'Epoch Test : {i} , Loss : {losses:.3f} , Acc : {(total_acc_te / total_cound_te):.3f}')
            seduch.step()
        save(Models.state_dict() , 'GTNet.pth')
        
    def Predicts(self , Model_dicts:str , kata:str):
        token_kata = tokenizer(kata)
        if token_kata[len(token_kata)-1] in vocab : 
            Models = GTN(self.n_label , self.vocabs , 768 , 
                         0.01 , 768 , 12 , 1).to(self.Device) 
            Models.load_state_dict(load(Model_dicts))
            kata_token = self.textpip(kata)
            print(Tensor(kata_token).unsqueeze(0))
            predik = Models(Tensor(kata_token).unsqueeze(0))
            print(predik.argmax().detach().numpy())
            print(lb_Globals.inverse_transform([predik.argmax().detach().numpy()]))
            print(kata)
            return lb_Globals.inverse_transform([predik.argmax().detach().numpy()])
        else : 
            out = 'input terdapat kata yang tidak terdeteksi'
            print(out)
            return out
        
Autrain = AutoTrain('promthmodel/Datas/Train.csv' , 'promthmodel/Datas/Test.csv')
#MUlpro = multiprocessing.Process(target=Autrain.Train())
Autrain.Predicts('GTNet.pth' , 'saya Mohon tampilkan foto saya')