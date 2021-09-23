import os
import tqdm
import torch
import datetime
import json
import logging
import numpy as np
from parser import create_parser
from data import CATH_Loader
from model import Struct2Seq, GCA
from sklearn.metrics import confusion_matrix
from utils import *
import warnings
warnings.filterwarnings("ignore")


class Exp:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_id = self.args.model_type + '_' + self.args.feature_type + '_' + \
                         str(self.args.dropout) + '_' + \
                         datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') \
                            if self.args.debug == 0 else 'debug'
        self.experiment_dir = os.path.join(self.args.models_dir, self.model_id)
        output_message(self.experiment_dir)
        self._build_model()

    def _preparation(self):
        set_seed(self.args.seed)
        # mkdir for saving experiments & print args
        if not os.path.exists(self.experiment_dir): os.makedirs(self.experiment_dir)
        self._save_param()
        # create logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            filename=self.experiment_dir + '/log.log',
                            filemode='w', # 'a':append or 'w':write
                            format='%(asctime)s - %(message)s')
        output_message(print_namespace(self.args))
        # prepare data
        output_message('Loading CATH dataset')
        cath_loader = CATH_Loader(self.args.cath_data, self.args.cath_splits, self.args.batch_tokens)
        self.train_loader, self.val_loader, self.test_loader = cath_loader.get_loader()
        self.num_to_letter = self.train_loader.num_to_letter
        output_message('Training:{}, Validation:{}, Test:{}'.format( \
                len(self.train_loader.dataset),len(self.val_loader.dataset),len(self.test_loader.dataset)))

    def _build_model(self):
        if self.args.model_type == 'structTrans':
            self.model = Struct2Seq(
                node_features=self.args.hidden,
                edge_features=self.args.hidden, 
                hidden_dim=self.args.hidden,
                protein_features=self.args.feature_type,
                dropout=self.args.dropout,
                use_mpnn=False,
                k_neighbors=self.args.top_k
            ).to(self.device)
        elif self.args.model_type == 'structGNN':
            self.model = Struct2Seq(
                node_features=self.args.hidden,
                edge_features=self.args.hidden, 
                hidden_dim=self.args.hidden,
                protein_features=self.args.feature_type,
                dropout=self.args.dropout,
                use_mpnn=True,
                k_neighbors=self.args.top_k
            ).to(self.device)
        elif self.args.model_type == 'gca':
            self.model = GCA(
                node_features=self.args.hidden,
                edge_features=self.args.hidden, 
                hidden_dim=self.args.hidden,
                dropout=self.args.dropout,
                k_neighbors=self.args.top_k,
                num_encoder_layers=self.args.num_layers,
                is_attention=self.args.is_attention,
            ).to(self.device)
        # prepare for training
        assert self.args.model_type in ['structTrans', 'structGNN', 'gca']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        print('Number of parameters: {}'.format(sum([p.numel() for p in self.model.parameters()])))

    def _acquire_device(self):
        if self.args.gpu < 0:  device = torch.device('cpu'); print('Using CPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda'); print('Using GPU: ', device)
        return device

    def _save_param(self):
        model_param = os.path.join(self.experiment_dir, 'model_param.json')
        with open(model_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

    def train(self):
        best_path, best_val = None, np.inf
        lookup = self.num_to_letter
        for epoch in range(self.args.epochs):
            self.model.train()
            loss, perplexity, confusion = self.loop(self.train_loader)
            recovery = self.test_recovery(self.train_loader)
            path = os.path.join(self.experiment_dir, str(epoch) + '.pt')
            torch.save(self.model.state_dict(), path)
            output_message(f'EPOCH {epoch} TRAIN loss: {loss:.4f} perplexity: {perplexity:.4f} recovery: {recovery:.4f}')
            
            self.model.eval()
            with torch.no_grad():
                loss, perplexity, confusion = self.loop(self.val_loader, is_train=False)    
            recovery = self.test_recovery(self.val_loader)
            output_message(f'EPOCH {epoch} VAL loss: {loss:.4f} perplexity: {perplexity:.4f} recovery: {recovery:.4f}')
            
            if loss < best_val:  best_path, best_val = path, loss
            output_message(f'BEST {best_path} VAL loss: {best_val:.4f} perplexity: {np.exp(best_val):.4f} recovery: {recovery:.4f}')
            
        output_message(f"TESTING: loading from {best_path}")
        self.model.load_state_dict(torch.load(best_path))
        
        self.model.eval()
        with torch.no_grad():
            loss, perplexity, confusion = self.loop(self.test_loader, is_train=False)
            recovery = self.test_recovery(self.test_loader)
        output_message(f'TEST loss: {loss:.4f} perplexity: {perplexity:.4f} recovery: {recovery:.4f}')
        output_message(print_confusion(confusion, lookup=lookup))

    def test_perplexity(self, dataloader):
        confusion = np.zeros((20, 20))
        train_sum, train_weight = 0, 0
        t = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for batch in t:
                X, S, mask, lengths = featurize(batch, self.device)
                log_probs = self.model(X=X, S=S, L=lengths, mask=mask)
                loss, _ = loss_nll(S, log_probs, mask)
                train_sum += torch.sum(loss * mask).cpu().data.numpy()
                train_weight += torch.sum(mask).cpu().data.numpy()

                pred = torch.argmax(log_probs, dim=-1).detach().cpu().numpy()
                true = S.detach().cpu().numpy()
                t.set_description(desc='loss: ' + format(loss.mean().item(), '.4f'))

                for b in range(true.shape[0]):
                    confusion += confusion_matrix(true[b], pred[b], labels=range(20))
                
        train_loss = train_sum / train_weight
        perplexity = np.exp(train_loss)
        return perplexity

    def test_recovery(self, dataloader):
        recovery = []
        t = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for batch in t:
                X, S, mask, lengths = featurize(batch, self.device)
                sample = self.model.sample(X=X, L=lengths, mask=mask, temperature=0.1)
                recovery_ = sample.eq(S).float().mean().cpu().numpy()
                recovery.append(recovery_)
                t.set_description(desc='recovery: ' + format(recovery_, '.4f'))
        return np.median(recovery)

    def loop(self, dataloader, is_train=True):
        confusion = np.zeros((20, 20))
        t = tqdm.tqdm(dataloader)
        train_sum, train_weight = 0, 0
        for batch in t:
            # do augmentation?
            X, S, mask, lengths = featurize(batch, self.device)
            if is_train == True and self.args.augment == 1:
                augmented_X = _rand_translate(_rand_rotate(X))
                log_probs = self.model(X=augmented_X, S=S, L=lengths, mask=mask)
            else:
                X, S, mask, lengths = featurize(batch, self.device)
                log_probs = self.model(X=X, S=S, L=lengths, mask=mask)

            if is_train:
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask)
                loss_sum = loss_av_smoothed
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()

            loss, _ = loss_nll(S, log_probs, mask)
            train_sum += torch.sum(loss * mask).cpu().data.numpy()
            train_weight += torch.sum(mask).cpu().data.numpy()

            pred = torch.argmax(log_probs, dim=-1).detach().cpu().numpy()
            true = S.detach().cpu().numpy()
            for b in range(true.shape[0]):
                confusion += confusion_matrix(true[b], pred[b], labels=range(20))
            t.set_description(desc='loss: ' + format(loss.mean().item(), '.4f'))
            torch.cuda.empty_cache()

        train_loss = train_sum / train_weight
        return train_loss, np.exp(train_loss), confusion

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    exp._preparation()

    exp.train()