import os
import json

import torch
from torchvision import datasets, transforms

class ExpMgr:
    def __init__(self, logger):
        self.logger = logger
        self.expnum = self.logger.get_num_logs()
        # self.load_exps()
        print("Experiment Manager is initialized")
        print("Total number of exps: {}".format(self.expnum))
        
    def create_exp(self, args, exec_func):
        
        exp_id = self.expnum + 1
        arg_append = update_args(args, [('exp_id', exp_id)])

        log_head = {'exp_id': exp_id, 'exp_desc':args.exp_desc}
        
        self.logger.save_log(log_head)
        print("EXP {} is starting".format(exp_id))
        exec_func(arg_append)
        # try: 
        #     exec_func(arg_append)
        # except:
        #     print("Exception has occurred and the EXP {} is cancelled".format(exp_id))

class DataVersion:
    """
    version:
            version_id (int)
            filepath (str)
            mnist_train: 
                subset info
                {'type': 'range', 'start_idx', 'end_idx' }
                {'type': 'map', 'mapper': list of range }
            mnist_test:
                subset info (range)
            exp_id (int)
            desc (text)
    """
    def __init__(self, ver_dict):
        self.vid = ver_dict['version_id']
        self.filepath = ver_dict['filepath']
        self.ver_dict = ver_dict
        
        
class MNIST_Mgr:

    def __init__(self):
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
        self.dataset_test = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)

        self.cur_vid = 0
        self.ver_dir = './versions/'
        # Load version info (for now: from './versions/')

        self.versions = []
        self.load_versions()
        
        print("MNIST Dataset Manager is initialized")
        print("total versions: {}".format(len(self.versions)))
        print("current vid: {}".format(self.cur_vid))

    def load_versions(self):
        files = None
        try:
            files = os.listdir(self.ver_dir)
        except:
            os.makedirs(self.ver_dir)
            return
        if len(files)==0:
            return
        files = [file for file in files if file[-3:]=='.dv']
        files.sort(key=lambda x:int(x[:-3]),reverse=False)
        for file in files:
            with open(os.path.join(self.ver_dir, file)) as f:
                ver_dict = json.load(f)
                self.versions.append(DataVersion(ver_dict))
        assert self.check_vid_validity()==True
            
        self.cur_vid = self.versions[-1].vid
        
    def check_vid_validity(self):
        if len(set(self.versions))!=len(self.versions):
            return False
        else:
            return True

    def create_version(self, ver_dict):
        ver_dict['version_id'] = self.cur_vid + 1
        filename = str(ver_dict['version_id']) + '.dv'
        ver_dict['filepath'] = os.path.join(self.ver_dir, filename)
        
        self.versions.append(DataVersion(ver_dict))
        with open(ver_dict['filepath'], 'w') as f:
            json.dump(ver_dict, f)
        self.cur_vid += 1
        return self.cur_vid

    # def append_version(self, vid, update_dict):
    #     ex_ver = self.get_by_vid(vid)
    #     if update_dict['mnist_train']['type'] == 'map':
    #         ex_ver['mnist_train']['mapper'].append(update_dict[])
    #     elif  update_dict['type'] == 'range':
    
    def get_train(self, timestep=None, stepsize=None):
        if timestep==None and stepsize==None:
            return self.dataset_train
        elif timestep!=None and stepsize!=None:
            return torch.utils.data.Subset(self.dataset_train, range((timestep-1)*stepsize, timestep*stepsize))
            
    def get_test(self, timestep=None, stepsize=None):
        if timestep==None and stepsize==None:
            return self.dataset_test
        elif timestep!=None and stepsize!=None:
            return torch.utils.data.Subset(self.dataset_test, range((timestep-1)*stepsize, timestep*stepsize))
    def get_by_vid(self, vid):
        if len(self.versions) < vid:
            return False
        key = vid - 1
        while(True):
            cursor = self.versions[key].vid
            if (cursor == vid):
                break
            elif cursor > vid:
                key -= 1
            elif cursor < vid:
                return None
    
        return self.get_versioned(self.versions[key])
        
    def get_versioned(self, version):
        subset_train = None
        subset_test = None
        concats = []
        ver = version.ver_dict
        if ver['mnist_train'] != None:
            train = ver['mnist_train'] 
            if train['type'] == 'range':
                subset_train = torch.utils.data.Subset(self.dataset_train, range(train['start_idx'], train['end_idx']))
            elif train['type'] == 'map':
                subset_train = torch.utils.data.Subset(self.dataset_train, train['mapper'])
            concats.append(subset_train)
        if ver['mnist_test'] != None:
            test = ver['mnist_test']
            if test['type'] == 'range':
                subset_test = torch.utils.data.Subset(self.dataset_test, range(test['start_idx'], test['end_idx']))
            elif test['type'] == 'map':
                subset_test = torch.utils.data.Subset(self.dataset_test, test['mapper'])
            concats.append(subset_test)
        return torch.utils.data.ConcatDataset(concats)

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        try:
            logs = os.listdir(log_dir)
            self.initLSN(logs)
        except FileNotFoundError:
            os.mkdir(log_dir)

        print("Logger is initialized")
        print("total logs: {}, cur_LSN: {}".format(self.get_num_logs(), self.cur_LSN))
            
    def initLSN(self, logs):
        # Check LSN integrity
        logs_removed_ext = [x.split('.')[0] for x in logs]
        LSNs = []
        LOGNAME_FAIL = False
        LSN_Integrity = True
        for log in logs_removed_ext:
            try:
                LSNs.append(int(log))
            except:
                self.LOGNAME_FAIL = True
        if len(LSNs) == 0:
            self.cur_LSN = -1
            return
        if len(set(LSNs)) == len(LSNs):
            self.cur_LSN = max(LSNs)
        else:
            self.LSN_Integrity = False
            self.cur_LSN = max(LSNs)
            # TODO: resolve conflicts
    
    def save_log(self, content):
        files = os.listdir(self.log_dir)
        self.cur_LSN = self.cur_LSN + 1
        with open(os.path.join(self.log_dir, (str(self.cur_LSN)+'.log')), 'w') as f:
            json.dump(content, f)
    
    def append_log(self, content):
        with open(os.path.join(self.log_dir, (str(self.cur_LSN)+'.log')), 'a') as f:
            f.write(',')
            json.dump(content, f)

    def load_logs(self):
        ret = []
        logs = [x for x in os.listdir(self.log_dir) if x[-4:]=='.log']
        for log in logs:
            with open(os.path.join('./log/',log)) as jsonfile:
                ret.append(json.load(jsonfile))
        return ret

    def get_num_logs(self):
        logs = [x for x in os.listdir(self.log_dir) if x[-4:]=='.log']
        return len(logs)


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def update_args(args, kv_list):
    dic = vars(args)
    for k, v in kv_list:
        dic[k] = v
    return Bunch(dic)

def create_ckpt_id(args):
    # Model Checkpoint Filename: <arch_name>_<exp_num>_<timestep>.pt
    output_dir = args.output_dir
    arch = args.model_arch
    exp_id = str(args.exp_id)
    timestep = str(args.timestep)
    ckpt_name = output_dir+'_'+arch+'_'+exp_id+'_'+timestep+'.pt'
    return ckpt_name

def get_ckpt_info(model_arch):
    # check designated model_architecture's trained history
    modelckpts = os.listdir('./models/')
    matched = []
    for model in modelckpts:
        if model.find(model_arch) == 0:
            matched.append(model)
    return matched

def get_latest_model(models, timestep):
    latest_ver = -1
    latest_model = None
    for model in models:
        ver = -1
        try:
            ver = int(model.split('.')[0].split('_')[-1])
        except ValueError:
            ver = 0
            if latest_ver < ver:
                latest_ver = ver
                latest_model = model
            pass
        if ver < timestep and latest_ver < ver:
            latest_ver = ver
            latest_model  = model
    return latest_ver, latest_model

