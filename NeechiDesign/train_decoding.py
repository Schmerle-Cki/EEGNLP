import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, \
    BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification

from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from config import get_config
from loss import ContrastiveLoss, cal_loss

CSLoss = None


def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, alpha=1, beta=1, cca_weight=1, wd_weight=1,
                checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt',
                checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt',
                output_dev_path='./results/dev_results.json'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    # NOTE: save visualized dev middle outputs
    os.makedirs(os.path.dirname(output_dev_path), exist_ok=True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    dev_output = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        strID = 0
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for (input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, raw_eeg, sentence, subject) in tqdm(dataloaders[phase]):
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                seq2seqLMoutput, eeg_embedding, words_embedding = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                        target_ids_batch)

                """calculate loss"""
                loss = seq2seqLMoutput.loss  # use the BART language modeling loss
                # print('Seq2seq Loss: ', loss.item())

                """align eeg and text for the training phase"""
                if phase == 'train':
                    words_embedding = words_embedding.mean(dim=-1)
                    eeg_embedding = eeg_embedding.mean(dim=-1)
                    # TODO: Criterion Loss within Emedding
                    # criterion = nn.MSELoss()
                #     loss += - alpha * criterion(eeg_embedding, words_embedding)
                #     # print('Criterion Loss: ', loss.item())
                    # TODO: CSL Loss
                    loss += beta * CSLoss(eeg_embedding, words_embedding)
                    # print('Contrastive Loss: ', loss.item())
                    # TODO: Wasserstein Distance --- remove 'Canonical Correlation Analysis +'
                    loss += cal_loss(cca_weight, wd_weight, text_embed=words_embedding, eeg_embed=eeg_embedding)
                    # print('WD Loss: ', loss.item())

                # """check prediction, instance 0 of each batch"""
                # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size(), ',target_mask size', target_mask_batch.size())
                # logits = logits.permute(0,2,1)
                # for idx in [0]:
                #     print(f'-- instance {idx} --')
                #     # print('permuted logits size:', logits.size())
                #     probs = logits[idx].softmax(dim = 1)
                #     # print('probs size:', probs.size())
                #     values, predictions = probs.topk(1)
                #     # print('predictions before squeeze:',predictions.size())
                #     predictions = torch.squeeze(predictions)
                #     # print('predictions:',predictions)
                #     # print('target mask:', target_mask_batch[idx])
                #     # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
                #     print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()
                # NOTE: visualized decoded results
                else:
                    logits = seq2seqLMoutput.logits  # bs*seq_len*voc_sz
                    probs = logits.softmax(dim=-1)
                    values, predictions = probs.topk(1)
                    predictions = torch.squeeze(predictions, dim=-1)
                    # print(f'predictions:{predictions} predictions shape:{predictions.shape}')
                    predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True, )
                    # print(f'predicted_string:{predicted_string}')

                    # start = predicted_string.find("[CLS]") + len("[CLS]")
                    # end = predicted_string.find("[SEP]")
                    # predicted_string = predicted_string[start:end]
                    # predicted_string=merge_consecutive_duplicates(predicted_string,'。')
                    # predictions=tokenizer.encode(predicted_string)
                    # TODO: Write down dev results
                    # for str_id in range(len(predicted_string)):
                    #     dev_item = {}
                    #     dev_item['Epoch'] = epoch
                    #     dev_item['StrID'] = strID
                    #     dev_item['Predicted'] = predicted_string[str_id]
                    #     dev_item['True'] = sentence[str_id]
                    #     dev_item['Subject'] = subject[str_id]
                    #     dev_item['EEGEmbedding'] = eeg_embedding.detach().cpu().tolist()
                    #     dev_item['TextEmbedding'] = words_embedding.detach().cpu().tolist()
                    #     dev_item['PredictedTokens'] = predictions[str_id].detach().cpu().tolist()
                    #     dev_item['SrcTokens'] = target_ids[str_id].detach().cpu().tolist()

                    #     dev_output.append(dev_item)
                    #     strID += 1
                    # json.dump(dev_output, open(output_dev_path, 'w'), indent=4)
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


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    home_directory = os.path.expanduser("../")
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_subj' # 'unique_sent'
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']

    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']

    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    csl_weight = args['contrastive_weight']
    wd_weight = args['wd_weight']

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4
        
    print(f'[INFO]using model: {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_csl{csl_weight}_wd{wd_weight}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_csl{csl_weight}_wd{wd_weight}_{step1_lr}_{step2_lr}_{dataset_setting}'
    
    if use_random_init:
        save_name = 'randinit_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')
    dev_path = os.path.join(save_path, f'dev-output_{save_name}.json')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug] subject_choice = using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu" 
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = 'datasets/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        dataset_path_task1=os.path.join(home_directory,dataset_path_task1)
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = 'datasets/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        dataset_path_task2=os.path.join(home_directory,dataset_path_task2)
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = 'datasets/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        dataset_path_task3=os.path.join(home_directory,dataset_path_task3)
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = 'datasets/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        dataset_path_taskNRv2=os.path.join(home_directory,dataset_path_taskNRv2)
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    
    tokenizer = BartTokenizer.from_pretrained('./bart-large')

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    if use_random_init:
        config = BartConfig.from_pretrained('./bart-large')
        pretrained = BartForConditionalGeneration(config)
    else:
        pretrained = BartForConditionalGeneration.from_pretrained('./bart-large')
    model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    model.to(device)
    
    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False
 

    if skip_step_one:
        if load_step1_checkpoint:
            # TODO: Nicki revised
            stepone_checkpoint = './checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b32_20_30_5e-05_5e-07_unique_subj_saveLastFail.pt' # 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, alpha=args['criterion_weight'], beta=args['contrastive_weight'], cca_weight=args['cca_weight'], wd_weight=args['wd_weight'])

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()
    CSLoss = ContrastiveLoss(temperature=args['temperature'], device=device)
    
    print()
    print('=== start Step2 training ... ===')
    # TODO: print training layers
    show_require_grad_layers(model)
    
    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, output_dev_path = dev_path, alpha=args['criterion_weight'], beta=csl_weight, cca_weight=args['cca_weight'], wd_weight=wd_weight)

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))
