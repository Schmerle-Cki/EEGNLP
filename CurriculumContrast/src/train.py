"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

import copy
import os
import pickle
import time
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from CSCL import CSCL
from dataset import ZuCo, build_CSCL_maps
from model import BrainTranslatorPreEncoder, BrainTranslator
from utils.set_seed import set_seed
from metrics import compute_metrics


def train_BrainTranslator(
        model, dataloaders, dataset_sizes, loss_fn, optimizer, epochs, device, tokenizer, scheduler,
        checkpoint_path_best='./checkpoints/decoding/best/translator.pt',
        checkpoint_path_last='./checkpoints/decoding/last/translator.pt'
        ):
    # FIXME: Nicki
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            # input_sample['input_embeddings'],
            # input_sample['seq_len'],
            # input_sample['input_attn_mask'],
            # input_sample['input_attn_mask_invert'],
            # input_sample['target_ids'],
            # input_sample['target_mask'],
            # input_sample['subject'],
            # input_sample['sentence']
            for batch, (input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, subject, sentence) in enumerate(dataloaders[phase]):
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                seq2seqLMoutput = model(input_embeddings_batch, input_mask_invert_batch, input_masks_batch, 
                                        target_ids_batch)

                """calculate loss"""
                # NOTE: my criterion not used
                loss = seq2seqLMoutput.loss  # use the BART language modeling loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_CSCL(
        model, dataloaders, cscl, loss_fn, optimizer, epochs, device, wnb
        ):
    checkpoint_path_best='./checkpoints/decoding/best/decoding.pt'
    checkpoint_path_last='./checkpoints/decoding/last/decoding.pt'
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for level in range(3):
        best_loss = 100000000000

        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            for phase in ['dev', 'train']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                loader = dataloaders[phase]

                for batch, (EEG, _, _, _, _, _, subject, sentence) in enumerate(loader):

                    E, E_pos, E_neg, mask, mask_pos, mask_neg = cscl[phase].get_triplet(
                        EEG, subject, sentence, level
                        ) # E = EEG

                    with torch.set_grad_enabled(phase == 'train'):
                        mask_triplet = torch.vstack((mask, mask_pos, mask_neg)).to(device)
                        out = model(
                            torch.vstack((E, E_pos, E_neg)).to(device),
                            mask_triplet
                            )
                        # out = torch.mean(out, dim=1)
                        # invert mask and average pre-encoder outputs
                        mask_triplet = abs(mask_triplet-1).unsqueeze(-1)
                        out = (out * mask_triplet).sum(1) / mask_triplet.sum(1)

                        h = out[:E.size(0), :]
                        h_pos = out[E.size(0):2*E.size(0), :]
                        h_neg = out[2*E.size(0):, :]
                        # h = torch.mean(out, dim=1)
                        # h = h.view(-1, 3, h.shape[-1])

                        T = 1
                        num = torch.exp(F.cosine_similarity(h, h_pos, dim=1)/T)
                        denom = torch.empty_like(num, device=num.device)
                        for j in range(E.size(0)):
                            denomjj = 0
                            for jj in range(E.size(0)):
                                denomjj += torch.exp(F.cosine_similarity(h[j, :], h_pos[jj, :], dim=0)/T)
                                denomjj += torch.exp(F.cosine_similarity(h[j, :], h_neg[jj, :], dim=0)/T)
                            denom[j] = denomjj

                        # num = torch.exp(
                        #     F.cosine_similarity(h[:, 0, :], h[:, 1, :], dim=1) / T
                        #     )
                        # denom = torch.empty_like(num, device=num.device)
                        # for j in range(E.size(0)):
                        #     denomjj = 0
                        #     for jj in range(E.size(0)):
                        #         denomjj += torch.exp(F.cosine_similarity(h[j, 0, :], h[jj, 1, :], dim=0) / T)
                        #         denomjj += torch.exp(F.cosine_similarity(h[j, 0, :], h[jj, 2, :], dim=0) / T)
                        #     denom[j] = denomjj

                        loss = -torch.log(num / denom).mean()
                        print(f'{epoch}.{batch} {phase} Loss: {loss:.4f}')
                        # print(f'{epoch}.{batch} {phase} Loss: {loss:.4e}')

                        if phase == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()

                        if wnb:
                            wandb.log({f"{phase} batch loss": loss.item()})

                        running_loss += loss.item()

                epoch_loss = running_loss / len(loader)
                print(f'{phase} Loss: {epoch_loss:.4f}')
                # print(f'{phase} Loss: {epoch_loss:.4e}')

                if wnb:
                    wandb.log({f"{phase} epoch loss": epoch_loss})

                if phase == 'dev' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), checkpoint_path_best)
                    # TODO: COMPUTE METRICS

            print()

    time_elapsed = time.time() - since
    print(f'CSCL Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')
    torch.save(model.state_dict(), checkpoint_path_last)

    model.load_state_dict(best_model_wts)
    return model

# FIXME: Nicki
def evaluate(loader, device, tokenizer, model, output_all_results_path):
    model.eval()

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []

    with open(output_all_results_path, 'w') as f:
        for batch, (input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, subject, sentence) in enumerate(loader):
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
            # print('target string:', target_string)

            # add to list for later calculate bleu metric
            # target_tokens_list.append([target_tokens])
            target_string_list.extend(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
            seq2seqLMoutput = model(input_embeddings_batch, input_mask_invert_batch, input_masks_batch, target_ids_batch)          
            
            logits = seq2seqLMoutput.logits  # bs*seq_len*voc_sz
            probs = logits.softmax(dim=-1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions, dim=-1)
            # print('#-# ' * 20, '\n', predictions)
            predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True, )
            # print(f'predicted_string:{predicted_string}')

            # start = predicted_string.find("[CLS]") + len("[CLS]")
            # end = predicted_string.find("[SEP]")
            # predicted_string = predicted_string[start:end]
            # predicted_string=merge_consecutive_duplicates(predicted_string,'。')
            # predictions=tokenizer.encode(predicted_string)
            for str_id in range(len(target_string)):
                f.write(f'start################################################\n')
                f.write(f'Predicted: {predicted_string[str_id]}\n')
                f.write(f'True: {target_string[str_id]}\n')
                f.write(f'end################################################\n\n\n')
            # convert to int list
            # predictions = predictions.tolist()
            # truncated_prediction = []
            # for t in predictions:
            #     if t != tokenizer.eos_token_id:
            #         truncated_prediction.append(t)
            #     else:
            #         break
            # pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # pred_tokens_list.append(pred_tokens)
            pred_string_list.extend(predicted_string)
            # sample_count += 1
            # print('predicted tokens:',pred_tokens)
            # print('predicted string:',predicted_string)
            # print('-' * 100)
    # print(f'pred_string_list:{pred_string_list}')
    # print(f'target_string_list:{target_string_list}')
    metrics_results=compute_metrics(pred_string_list,target_string_list)
    print(metrics_results)
    print(output_all_results_path)
    output_all_metrics_results_path = output_all_results_path.replace('txt', 'json')
    print(output_all_metrics_results_path)
    with open(output_all_metrics_results_path, "w") as json_file:
        json.dump(metrics_results, json_file, indent=4, ensure_ascii=False)

def main():
    cfg = {
        'seed': 312,
        'subject_choice': 'ALL',
        'eeg_type_choice': 'GD',
        'bands_choice': 'ALL',
        'dataset_setting': 'unique_subj', # 'unique_sent',
        'batch_size': 1,
        'shuffle': False,
        'input_dim': 840,
        'num_layers': 1,  # 6
        'nhead': 1,  # 8
        'dim_pre_encoder': 2048,
        'dim_s2s': 1024,
        'dropout': 0,
        'T': 5e-6,
        'lr_pre': 1e-6,
        'epochs_pre': 5,
        'lr': 1e-6,
        'epochs': 5,
        'wandb': False
        }

    if cfg['wandb']:
        wandb.init(project='CSCL', reinit=True, config=cfg)

    set_seed(cfg['seed'])
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # data and dataloaders
    whole_dataset_dicts = []

    dataset_path_task1 = os.path.join(
        '../../', 'datasets', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )

    whole_dataset_dicts = []
    for t in [dataset_path_task1]:
        with open(t, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    train_set = ZuCo(
        whole_dataset_dicts,
        'train',
        BartTokenizer.from_pretrained('../../EEG2Text_new/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    dev_set = ZuCo(
        whole_dataset_dicts,
        'dev',
        BartTokenizer.from_pretrained('../../EEG2Text_new/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    dev_loader = DataLoader(
        dev_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    # FIXME: Nicki
    test_set = ZuCo(
        whole_dataset_dicts,
        'test',
        BartTokenizer.from_pretrained('../../EEG2Text_new/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    test_loader = DataLoader(
        test_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    tokenizer = BartTokenizer.from_pretrained('../../EEG2Text_new/bart-large')

    dataloaders = {'train': train_loader, 'dev': dev_loader}
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('#'*10, 'DataSet Sizes =', dataset_sizes)

    # train pre-encoder with CSCL
    model = BrainTranslatorPreEncoder(
        input_dim=cfg['input_dim'],
        num_layers=cfg['num_layers'],
        nhead=cfg['nhead'],
        dim_pre_encoder=cfg['dim_pre_encoder'],
        dim_s2s=cfg['dim_s2s'],
        dropout=cfg['dropout']
        ).to(device)

    if cfg['wandb']:
        wandb.watch(model, log='all')

    # TODO: BUG_EXIST_WHEN_USING_'unique_subj': for dev_set, all sentences come from one person, so the positive_pair would be none-pairs! __ Return itself!
    fs, fp, S = build_CSCL_maps(train_set)
    cscl_train = CSCL(fs, fp, S)

    fs, fp, S = build_CSCL_maps(dev_set)
    cscl_dev = CSCL(fs, fp, S)

    cscl = {'train': cscl_train, 'dev': cscl_dev}

    loss_fn = cfg['T']  # TODO
    optimizer = optim.Adam(params=model.parameters(), lr=cfg['lr_pre'])

    # TODO: with exist training parameters
    # model = train_CSCL(
    #     model, dataloaders, cscl, loss_fn, optimizer, cfg['epochs_pre'], device, cfg['wandb']
    #     )
    # exit(20240725)
    model.load_state_dict(torch.load('./checkpoints/decoding/best/decoding.pt'))

    
    model = BrainTranslator(
        model,
        BartForConditionalGeneration.from_pretrained('../../EEG2Text_new/bart-large'),
    ).to(device)

    # TODO: train BrainTranslator by Nicki
    loss_fn = None
    optimizer = optim.Adam(params=model.parameters(), lr=cfg['lr'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model = train_BrainTranslator(
        model, dataloaders, dataset_sizes, loss_fn, optimizer, cfg['epochs'], device, tokenizer, scheduler
        )    

    evaluate(test_loader, device, tokenizer, model, './results/decoding_results.txt')



if __name__ == "__main__":
    main()
