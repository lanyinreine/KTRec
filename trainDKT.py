import argparse
import os
import time

import torch
from torch.nn import BCELoss
from torch.nn.functional import nll_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from KTModel.DataLoader import MyDataset2, DatasetRetrieval, DatasetRec
from KTModel.utils import collate_fn, collate_co, collate_rec
from KTModel.utils import set_random_seed, load_model, evaluate_utils


def main(args: argparse.Namespace):
    print(args)
    set_random_seed(args.rand_seed)
    # Dataset
    dataset = DatasetRec if args.forRec else (MyDataset2 if not args.retrieval else DatasetRetrieval)
    dataset = dataset(os.path.join(args.data_dir, args.dataset))
    args.feat_nums = dataset.feats_num
    args.user_nums = dataset.users_num
    if args.forRec:
        args.output_size = dataset.feats_num
        args.without_label = True
    collate = collate_rec if args.forRec else (collate_fn if not args.retrieval else collate_co)
    dataloader = DataLoader(dataset, args.batch_size, True, num_workers=8, collate_fn=collate)
    # Model
    model = load_model(args).to(args.device)

    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.load_model:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"Load Model From {model_path}")
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    criterion = (lambda y_, y: nll_loss(torch.log(y_ + 1e-9), y)) if args.forRec else BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=args.decay_step, verbose=True,
                                  min_lr=args.min_lr)
    best_val_auc = 0
    print('-' * 20 + "Training Start" + '-' * 20)
    for epoch in range(args.num_epochs):
        avg_time = 0
        train_eval = [torch.tensor([]), torch.tensor([])]
        dataset.change_mode('train')
        model.train()
        for i, data in enumerate(tqdm(dataloader)):
            t0 = time.time()
            data = [_.to(args.device) for _ in data]
            if args.retrieval:
                users, feats, y = data[0], data[1:-1], data[-1]
            else:
                users, feats, y = data
            y_ = model(feats).squeeze(-1)
            loss = criterion(y_, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_eval[0] = torch.cat([train_eval[0], y.detach().cpu()])
            train_eval[1] = torch.cat([train_eval[1], y_.detach().cpu()])
            if train_eval[0].shape[0] > 10000:
                train_eval[0] = train_eval[0][-10000:]
                train_eval[1] = train_eval[1][-10000:]
            _, acc, auc = evaluate_utils(train_eval[1], train_eval[0])
            scheduler.step(auc)
            avg_time += time.time() - t0
            print(f'Epoch:{epoch}\tbatch:{i}\tavg_time:{avg_time / (i + 1):.4f}\t'
                  f'loss:{loss:.4f}\tacc:{acc:.4f}\tauc:{auc:.4f}')

        print('-' * 20 + "Validating Start" + '-' * 20)
        val_eval = [[], []]
        dataset.change_mode('valid')
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(tqdm(dataloader)):
                try:
                    y = data[-1]
                    data = [_.to(args.device) for _ in data[:-1]]
                    if args.retrieval:
                        users, feats = data[0], data[1:]
                    else:
                        users, feats = data
                    y_ = model(feats).squeeze(-1)
                    y_ = y_.cpu()
                    val_eval[0].append(y)
                    val_eval[1].append(y_)
                    if j >= args.valid_step:
                        break
                except Exception as e:
                    print(e)
                    continue
            val_eval = [torch.cat(_) for _ in val_eval]
            loss, acc, auc = evaluate_utils(val_eval[1], val_eval[0], criterion)
            print(f"Validating loss:{loss:.4f} acc:{acc:.4f} auc:{auc:.4f}")
        if auc >= best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), model_path)
            print("New best result Saved!")
        print(f"Best Auc Now:{best_val_auc:.4f}")

    print('-' * 20 + "Testing Start" + '-' * 20)
    test_eval = [[], []]
    dataset.change_mode('test')
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    with torch.no_grad():
        for k, data in enumerate(tqdm(dataloader)):
            try:
                y = data[-1]
                data = [_.to(args.device) for _ in data[:-1]]
                if args.retrieval:
                    users, feats = data[0], data[1:]
                else:
                    users, feats = data
                y_ = model(feats).squeeze(-1)
                y_ = y_.cpu()
                test_eval[0].append(y)
                test_eval[1].append(y_)
            except Exception as e:
                print(e)
                continue
        test_eval = [torch.cat(_) for _ in test_eval]
        print(test_eval[1])
        print(test_eval[0])
        loss, acc, auc = evaluate_utils(test_eval[1], test_eval[0], criterion)
        print(f"Testing loss:{loss:.4f} acc:{acc:.4f} auc:{auc:.4f}")


if __name__ == '__main__':
    from KTModel.Configure import get_exp_configure

    torch.set_num_threads(6)
    parser = argparse.ArgumentParser(description='Learning Path Recommendation')
    parser.add_argument('-m', '--model', type=str, choices=['DKT', 'DKVMN', 'Transformer', 'CoKT', 'GRU4Rec'],
                        default='DKT', help="Model to use")
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['assist09', 'assist09DKT', 'junyi', 'junyiDKT', 'assist15', 'assist15DKT', 'ednet'],
                        default='assist09', help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--without_label', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--valid_step', type=int, default=100)
    parser.add_argument("--postfix", type=str, default="",
                        help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    args_ = parser.parse_args()
    # Get experiment configuration
    exp_configure = get_exp_configure(args_.model)
    args_ = argparse.Namespace(**vars(args_), **exp_configure)

    args_.exp_name = '_'.join([args_.model, args_.dataset])
    if args_.postfix != '':
        args_.exp_name += '_' + args_.postfix

    device_name = 'cpu' if args_.cuda < 0 else f'cuda:{args_.cuda}'
    args_.device = torch.device(device_name)
    main(args_)
