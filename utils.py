import fastai
from fastai.vision import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler, SequentialSampler, RandomSampler
import sklearn


# Modification to ImageDataBunch to allow to give a list of custome samplers.
class ImageDataBunch(ImageDataBunch):
    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
               val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
               device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, samplers=None, **dl_kwargs)->'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs` and optionally a list of samplers."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        if samplers is None: samplers = [RandomSampler] + 3*[SequentialSampler]
        dls = [DataLoader(d, b, sampler=s(d, bs=b), num_workers=num_workers, **dl_kwargs) for d,b,s in
               zip(datasets, (bs,val_bs,val_bs,val_bs), samplers) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    

class ImageList(ImageList):
    _bunch = ImageDataBunch
    

class SequentialSampler(SequentialSampler):
    def __init__(self, data_source, **kwargs):
        self.data_source = data_source
        

class RandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, **kwargs):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        

class FixedLenRandomSampler(RandomSampler):
    """Sample epochs with a fixed length"""
    def __init__(self, data_source, bs, epoch_size, *args, **kwargs):
        super().__init__(data_source)
        self.epoch_size = epoch_size*bs
    
    def __iter__(self):
        return iter(np.random.choice(range(len(self.data_source)), size=len(self), replace=True).tolist())
    
    def __len__(self):
        return self.epoch_size


def load_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` cropped and resized."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        if Path(fn).parent.name == 'train':
            ind = train_df.loc[Path(fn).name, 'ind']
            x = X_train[ind]
        else:
            ind = test_df.loc[Path(fn).name, 'ind']
            x = X_test[ind]
        _, time_dim = x.shape
        if time_dim - base_dim > 0:
            crop_x = np.random.randint(0, time_dim - base_dim)
            x = x[:, crop_x:crop_x+base_dim]
        x = PIL.Image.fromarray(x).resize((SZ,SZ)).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)


def load_image_tta(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None, flip=False, vert=False, step=128)->Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        if Path(fn).parent.name == 'train':
            ind = train_df.loc[Path(fn).name, 'ind']
            x = X_train[ind]
        else:
            ind = test_df.loc[Path(fn).name, 'ind']
            x = X_test[ind]
        if flip: x = np.fliplr(x)
        if vert: x = np.flipud(x)
        _, time_dim = x.shape
        xb = []
        for n in range(0, max(1, time_dim-base_dim), step):
            x0 = PIL.Image.fromarray(x[:,n:n+base_dim]).resize((SZ,SZ)).convert(convert_mode)
            if after_open: x0 = after_open(x0)
            x0 = pil2tensor(x0, np.float32)
            if div: x0.div_(255)
            x0 = normalize(x0, mean=tensor([0.2932, 0.2932, 0.2932]), std=tensor([0.2556, 0.2556, 0.2556]))
            xb.append(x0[None])
        xb = torch.cat(xb, dim=0)
    return xb


class ImageListMemory(ImageList):
    """ImageList that load images from memory using load_image function"""
    def __init__(self, *args, convert_mode='L', after_open:Callable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_mode,self.after_open = convert_mode,after_open
        self.copy_new.append('convert_mode')
        self.c,self.sizes = 1,{}

    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return load_image(fn, convert_mode=self.convert_mode, after_open=self.after_open)


def _cutout(x, n_holes:uniform_int=1, length:uniform_int=40):
    "Cut out `n_holes` number of rectangular bands of size `length` in image at random locations."
    h,w = x.shape[1:]
    for n in range(n_holes):
        h_y = np.random.randint(0, h)
        h_x = np.random.randint(0, w)
        y1 = int(np.clip(h_y - length / 2, 0, h))
        y2 = int(np.clip(h_y + length / 2, 0, h))
        x1 = int(np.clip(h_x - length / 2, 0, w))
        x2 = int(np.clip(h_x + length / 2, 0, w))
        x[:, y1:y2, :] = 0
        x[:, :, x1:x2] = 0
    return x
cutout2 = TfmPixel(_cutout, )


class BCELoss(nn.Module):
    def __init__(self, reduce=False):
        super().__init__()
        self.reduce = reduce

    def forward(self, logit, target):
        target = target.float()
        loss = nn.BCEWithLogitsLoss()(logit, target)
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if not self.reduce:
            return loss
        else:
            return loss.mean()


# Adapted from https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduce=False):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if not self.reduce:
            return loss
        else:
            return loss.mean()

 
def fbeta2(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res


class MixupBCELoss(BCELoss):
    def forward(self, x, y):
        if isinstance(y, dict):
            y0, y1, a = y['y0'], y['y1'], y['a']
            loss = a*super().forward(x, y0) + (1-a)*super().forward(x, y1)

            if f2cl is not None:
                # Removing samples with F2 score equal to f2cl
                fbs = fbeta2(x, y0*a.view(-1,1)+(1-a.view(-1,1))*y1)
                loss = loss[(fbs<f2cl).byte()]
        else:
            loss = super().forward(x, y)
        return 100*loss.mean()


class MixupFocalLoss(FocalLoss):
    def forward(self, x, y):
        if isinstance(y, dict):
            y0, y1, a = y['y0'], y['y1'], y['a']
            loss = a*super().forward(x, y0) + (1-a)*super().forward(x, y1)

            if f2cl is not None:
                # Removing samples with F2 score equal to f2cl
                fbs = fbeta2(x, y0*a.view(-1,1)+(1-a.view(-1,1))*y1)
                loss = loss[(fbs<f2cl).byte()]
        else:
            loss = super().forward(x, y)
        return loss.mean()


# Calculate the overall lwlrap using sklearn.metrics function.
def lwlrap(scores, truth):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    scores = scores.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0, 
      scores[nonzero_weight_sample_indices, :], 
      sample_weight=sample_weight[nonzero_weight_sample_indices])
    return tensor(overall_lwlrap)


class AudioMixup(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train:
            bs = last_input.size()[0]
            lambd = np.random.uniform(0, 0.5, bs)
            shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
            x1, y1 = last_input[shuffle], last_target[shuffle]
            a = tensor(lambd).float().view(-1, 1, 1, 1).to(last_input.device)
            last_input = a*last_input + (1-a)*x1
            last_target = {'y0':last_target, 'y1':y1, 'a':a.view(-1)}
            return {'last_input': last_input, 'last_target': last_target}
        

def get_preds_tta(learn, valid=True, flip=False, vert=False):
    with torch.no_grad():
        preds0 = []
        N = len(learn.data.valid_ds) if valid else len(learn.data.test_ds)
        for i in progress_bar(range(N), total=N):
            if valid:
                xb = load_image_tta(learn.data.valid_ds.items[i], flip=flip, vert=vert, step=base_dim) 
            else: 
                xb = load_image_tta(learn.data.test_ds.items[i], flip=flip, vert=vert, step=base_dim)
            out = learn.model(xb.cuda())
            out = out.sigmoid().max(0)[0]
            preds0.append(out[None].cpu())
        preds0 = torch.cat(preds0, dim=0)
        return preds0
    

def print_scores(name, preds, ys):
    print(f'{name} | F2={fbeta(preds, ys).item():.4f}; LWL={lwlrap(preds, ys).item():.4f}')