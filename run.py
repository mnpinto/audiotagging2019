import utils, preprocessing
from utils import * 
from preprocessing import *
from models import *
from sklearn.model_selection import KFold
import argparse

def main(path=None, model=None, base_dim=None, SZ=None, BS=None, lr=None,
         n_epochs=None, epoch_size=None, f2cl=None, fold_number=None,
         loss_name=None, csv_name=None, weights_file=None, working_path=None,
         max_processors=None, load_weights=None, kaggle=None, force=None):
    utils.base_dim = base_dim
    utils.SZ = SZ
    utils.f2cl = f2cl

    if loss_name == 'BCELoss':
        loss_func = MixupBCELoss() 
    elif loss_name == 'FocalLoss':
        loss_func = MixupFocalLoss()
    else: raise NotImplementedError('Choose BCELoss or FocalLoss for the loss_name.')

    # Processing curated train dataset
    if (not (path/'train_curated_png').is_dir() and not kaggle) or force:
        print('\nComputing mel spectrograms for the curated train dataset and saving as .png:')
        train_df = pd.read_csv(path/'train_curated.csv')
        preprocessing.path_source = path/'train_curated'
        preprocessing.path_save = path/'train_curated_png'
        preprocessing.path_save.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_processors) as e: 
            list(progress_bar(e.map(convert_wav_to_png, list(train_df.iterrows())), total=len(train_df)))

    # Processing noisy train dataset
    if (not (path/'train_noisy_png').is_dir() and not kaggle) or force:
        print('\nComputing mel spectrograms for the noisy train dataset and saving as .png:')
        train_df = pd.read_csv(path/'train_noisy.csv')
        preprocessing.path_source = path/'train_noisy'
        preprocessing.path_save = path/'train_noisy_png'
        preprocessing.path_save.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_processors) as e: 
            list(progress_bar(e.map(convert_wav_to_png, list(train_df.iterrows())), total=len(train_df)))

    # Processing test data
    print('\nComputing mel spectrograms for the test dataset:')
    path2 = Path('../input/freesound-audio-tagging-2019/') if kaggle else path
    test_df = pd.read_csv(path2/'sample_submission.csv')
    X_test = convert_wav_to_image(test_df, fold='test', source=path2/'test')
    test_df['ind'] = test_df.index
    test_df.set_index('fname', inplace=True)
    test_df['fname'] = test_df.index

    # Load indices of noisy data to use 
    path_idx = 'audiotagging2019/data' if kaggle else path
    good_noisy = pd.read_csv(path_idx/'good_idx.csv').idx.values

    # Create train dataframe and list of arrays
    print('\n\nLoading train data:')
    train_df = pd.read_csv(path2/'train_curated.csv')
    train_df.loc[:, 'fname'] = [f[:-4] for f in train_df.fname]
    train_noisy_df = pd.read_csv(path2/'train_noisy.csv').iloc[good_noisy]
    train_noisy_df.loc[:, 'fname'] = [f[:-4] for f in train_noisy_df.fname]
    data_path = Path('../input/audiotagging128/train_curated_128/data/') if kaggle else path
    png = '' if kaggle else '_png'
    X_train_curated = [np.array(PIL.Image.open(data_path/f'train_curated{png}/{fn}.png')) for fn in progress_bar(train_df.fname)]
    X_train_noisy = [np.array(PIL.Image.open(data_path/f'train_noisy{png}/{fn}.png')) for fn in progress_bar(train_noisy_df.fname)]
    train_df = pd.concat((train_df, train_noisy_df)).reset_index(drop=True)
    train_df['ind'] = train_df.index
    train_df.set_index('fname', inplace=True)
    train_df['fname'] = train_df.index
    X_train = [*X_train_curated, *X_train_noisy]
        
    # Flipped images and labels
    for o in progress_bar(X_train.copy()):
        X_train.append(np.fliplr(o))
    train_df_flip = train_df.copy()
    train_df_flip.loc[:, 'labels'] = [','.join([f'{o}_flip' for o in a.split(',')]) for a in train_df_flip.labels.values]
    train_df_flip.loc[:, 'ind'] = train_df_flip.ind + train_df_flip.ind.max() + 1
    train_df_flip.loc[:, 'fname'] = [f'{o}_flip' for o in train_df_flip.fname]
    train_df_flip = train_df_flip.set_index('fname', drop=False)
    train_df = pd.concat((train_df, train_df_flip))

    # Vertical flip
    for o in progress_bar(X_train.copy()):
        X_train.append(np.flipud(o))
    train_df_flip_vert = train_df.copy()
    train_df_flip_vert.loc[:, 'labels'] = [','.join([f'{o}_vert' for o in a.split(',')]) for a in train_df_flip_vert.labels.values]
    train_df_flip_vert.loc[:, 'ind'] = train_df_flip_vert.ind + train_df_flip_vert.ind.max() + 1
    train_df_flip_vert.loc[:, 'fname'] = [f'{o}_vert' for o in train_df_flip_vert.fname]
    train_df_flip_vert = train_df_flip_vert.set_index('fname', drop=False)
    train_df = pd.concat((train_df, train_df_flip_vert))

    utils.train_df = train_df
    utils.test_df = test_df
    utils.X_train = X_train
    utils.X_test = X_test

    if fold_number is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=534)

        # Validation indices 
        valid_idx = list(kf.split(list(range(len(train_df)//4))))[fold_number][1]
        valid_idx_flip = valid_idx +      1*len(train_df)//4
        valid_idx_vert = valid_idx +      2*len(train_df)//4
        valid_idx_flip_vert = valid_idx + 3*len(train_df)//4
        valid_idx = [*valid_idx, *valid_idx_flip, *valid_idx_vert, *valid_idx_flip_vert]

    # List of augmentations
    tfms = get_transforms(do_flip=False, max_rotate=0, max_zoom=1.5, max_warp=0, 
                          xtra_tfms=[cutout2(n_holes=(1, 4), length=(5, 20), p=0.75)])

    # ImageLists
    train = ImageListMemory.from_df(train_df, path=data_path, cols='fname', folder='train') 
    test = ImageListMemory.from_df(test_df, path=data_path, cols='fname', folder='test') 

    # Custom samplers
    train_sampler = partial(FixedLenRandomSampler, epoch_size=epoch_size)
    samplers = [train_sampler, SequentialSampler, SequentialSampler, SequentialSampler]

    # Create databunch
    if fold_number is None:
        train_split = train.split_none() 
    else:
        train_split = train.split_by_idx(valid_idx)
        
    data = (train_split.label_from_df(cols='labels', label_delim=',')
            .add_test(test)
            .transform(tfms, size=SZ)
            .databunch(samplers=samplers, path=working_path, bs=BS)
            .normalize([tensor([0.2932, 0.2932, 0.2932]), tensor([0.2556, 0.2556, 0.2556])]))

    # Create learner
    mod_name = inspect.getmodule(model).__name__
    if 'fastai.vision.models' in  mod_name or 'torchvision.models' in mod_name:
        learn = cnn_learner(data, model, pretrained=False, loss_func=loss_func, metrics=[fbeta, lwlrap], 
                            callback_fns=[AudioMixup])
    else:
        learn = Learner(data, model(c_out=data.c), loss_func=loss_func, metrics=[fbeta, lwlrap], 
                            callback_fns=[AudioMixup])
    learn.clip_grad = 1

    if load_weights is not None:
        print(f'\n\nLoading {load_weights}.pth weights.')
        learn.load(load_weights)

    # Train
    print('\nTraning the model:')
    learn.fit_one_cycle(n_epochs, slice(lr))
    learn.save(weights_file)
    print(f'\nModel weights save to {working_path/"models"/weights_file}.pth.')

    normal = tensor(['_flip' not in o and '_vert' not in o for o in learn.data.classes]).long()
    flip = tensor(['_flip' in o and '_vert' not in o for o in learn.data.classes]).long()
    vert = tensor(['_vert' in o and '_flip' not in o for o in learn.data.classes]).long()
    flip_vert = tensor(['_flip' in o and '_vert' in o for o in learn.data.classes]).long()
    assert sum(normal) == 80
    assert sum(flip) == 80
    assert sum(vert) == 80
    assert sum(flip_vert) == 80

    # Validate only if fold number is not None, otherwise all data has been used to train
    if fold_number is not None:
        print('\nComputing validation scores without TTA:')
        preds0, ys0 = learn.get_preds(ds_type=DatasetType.Valid)
        S = len(preds0)//4
        preds_normal, ys_normal = preds0[:1*S, normal.byte()], ys0[:1*S, normal.byte()]
        preds_flip, ys_flip = preds0[1*S:2*S, flip.byte()], ys0[1*S:2*S, flip.byte()]
        preds_vert, ys_vert = preds0[2*S:3*S, vert.byte()], ys0[2*S:3*S, vert.byte()]
        preds_flip_vert, ys_flip_vert = preds0[3*S:, flip_vert.byte()], ys0[3*S:, flip_vert.byte()]
        print_scores('Mix   ', preds0, ys0)
        print_scores('Normal', preds_normal, ys_normal)
        print_scores('Flip  ', preds_flip, ys_flip)
        print_scores('Vert  ', preds_vert, ys_vert)
        print_scores('FlVert', preds_flip_vert, ys_flip_vert)
        print_scores('Ensble', preds_normal.sigmoid() + preds_flip.sigmoid() + preds_vert.sigmoid() + preds_flip_vert.sigmoid(), ys_normal) 

        print('\nComputing validation scores with TTA:')
        preds0 = get_preds_tta(learn)
        S = len(preds0)//4
        preds_normal, ys_normal = preds0[:1*S, normal.byte()], ys0[:1*S, normal.byte()]
        preds_flip, ys_flip = preds0[1*S:2*S, flip.byte()], ys0[1*S:2*S, flip.byte()]
        preds_vert, ys_vert = preds0[2*S:3*S, vert.byte()], ys0[2*S:3*S, vert.byte()]
        preds_flip_vert, ys_flip_vert = preds0[3*S:, flip_vert.byte()], ys0[3*S:, flip_vert.byte()]
        print_scores('Mix   ', preds0, ys0)
        print_scores('Normal', preds_normal, ys_normal)
        print_scores('Flip  ', preds_flip, ys_flip)
        print_scores('Vert  ', preds_vert, ys_vert)
        print_scores('FlVert', preds_flip_vert, ys_flip_vert)
        preds_ens = torch.cat((preds_normal[None], preds_flip[None], preds_vert[None], preds_flip_vert[None]), dim=0)
        print_scores('Ensble', preds_ens.mean(0), ys_normal) 

    # Compute results for test set and generate submission csv.
    print('\nComputing predictions for test data:')
    _ = learn.get_preds(ds_type=DatasetType.Test)
    preds_normal = get_preds_tta(learn, valid=False)
    preds_flip = get_preds_tta(learn, valid=False, flip=True)
    preds_vert = get_preds_tta(learn, valid=False, vert=True)
    preds_flip_vert = get_preds_tta(learn, valid=False, flip=True, vert=True)
    preds_normal = preds_normal[:, normal.byte()]
    preds_flip = preds_flip[:, flip.byte()]
    preds_vert = preds_vert[:, vert.byte()]
    preds_flip_vert = preds_flip_vert[:, flip_vert.byte()]
    preds_all = (preds_normal + preds_flip + preds_vert + preds_flip_vert)/4

    classes = [c for c in learn.data.classes if '_flip' not in c and '_vert' not in c]
    assert len(classes) == 80

    for i, v in enumerate(classes):
        test_df[v] = preds_all[:, i]
    test_df.to_csv(working_path/csv_name, index=False)
    print(f'\n\nPredictions saved to {working_path/csv_name}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='data')
    arg('--working_path', type=str, default='.')
    arg('--base_dim', type=int, default=128)
    arg('--SZ', type=int, default=128)
    arg('--BS', type=int, default=64)
    arg('--lr', type=float, default=1e-2)
    arg('--n_epochs', type=int, default=80)
    arg('--epoch_size', type=int, default=1000)
    arg('--f2cl', type=int, default=1)
    arg('--fold_number', type=int, default=0)
    arg('--loss_name', type=str, default='BCELoss', choices=['BCELoss', 'FocalLoss'])
    arg('--csv_name', type=str, default='submission.csv')
    arg('--model', type=str, default='models.xresnet18')
    arg('--load_weights', type=str, default='')
    arg('--weights_file', type=str, default='stage-1')
    arg('--max_processors', type=int, default=8)
    arg('--kaggle', type=bool, default=False)
    arg('--force', type=bool, default=False)
    args = parser.parse_args()
    
    path = Path(args.path)
    model = eval(args.model)
    working_path = Path(args.working_path)

    fold_number = args.fold_number
    if fold_number == -1:
        fold_number = None

    load_weights = args.load_weights
    if load_weights == '':
        load_weights = None

    print('\nStarting run using the following configuration:')
    for arg in vars(args):
        print(f'{arg:14s}: {getattr(args, arg)}')

    main(path=path, model=model, working_path=working_path, base_dim=args.base_dim, SZ=args.SZ, BS=args.BS,
         lr=args.lr, n_epochs=args.n_epochs, epoch_size=args.epoch_size, f2cl=args.f2cl, fold_number=fold_number,
         loss_name=args.loss_name, csv_name=args.csv_name, weights_file=args.weights_file, max_processors=args.max_processors,
         load_weights=load_weights, kaggle=args.kaggle, force=args.force)