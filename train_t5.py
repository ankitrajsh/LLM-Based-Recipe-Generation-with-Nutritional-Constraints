from google.cloud import storage
from io import StringIO
import pandas as pd
import torch
import wandb
from torch import cuda
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from time import perf_counter
import argparse
import sys

def get_df_from_gcs_blob(blob, bucket='recipe-data-bucket'):
    # START: COPIED FROM https://github.com/googleapis/python-storage/blob/HEAD/samples/snippets/storage_fileio_write_read.py
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(blob)
    blob = blob.download_as_string()
    blob = blob.decode()
    blob = StringIO(blob)  #tranform bytes to string here
    df = pd.read_csv(blob)
    return df
    # END: COPIED FROM https://github.com/googleapis/python-storage/blob/HEAD/samples/snippets/storage_fileio_write_read.py

def save_model(path, epoch, mod, opt, running_loss):
   torch.save({
            'epoch': epoch,
            'model_state_dict': mod.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': running_loss,
            }, path)

# START: COPIED FROM https://www.kaggle.com/code/kreeshrajani/fine-tune-t5-for-conversational-model
class T5Dataset:
  def __init__(self, inps, outs, tokenizer, inp_max_len, out_max_len):   
    self.inps = inps
    self.outs = outs
    self.tokenizer = tokenizer
    self.input_max_len = inp_max_len
    self.output_max_len = out_max_len
  
  def __len__(self):                      # This method retrives the number of item from the dataset
    return len(self.inps)

  def __getitem__(self, item):             # This method retrieves the item at the specified index item. 
    inp = str(self.inps[item])
    out = str(self.outs[item])

    input_tokenize = self.tokenizer(      
            inp,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors="pt"
        )
    output_tokenize = self.tokenizer(
            out,
            add_special_tokens=True,
            max_length=self.output_max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors="pt"
            
        )
    

    input_ids = input_tokenize["input_ids"].flatten().to(dtype=torch.long)
    attention_mask = input_tokenize["attention_mask"].flatten().to(dtype=torch.long)
    output_ids = output_tokenize['input_ids'].flatten().to(dtype=torch.long)

    out = {
            'input': inp,      
            'target': out,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': output_ids
        }
        
    return out 
# END: COPIED FROM https://www.kaggle.com/code/kreeshrajani/fine-tune-t5-for-conversational-model

# START: PARTIALLY COPIED FROM https://github.com/Shivanandroy/T5-Finetuning-PyTorch
def train(tokenizer, model, device, loader, optimizer, fp16=True):
    losses = []
    if fp16: model.half()
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        losses.append(loss.item())
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

def test(tokenizer, model, device, loader, fp16=True):
    losses = []
    if fp16: model.half()
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            losses.append(loss.item())
            
            if _%10 == 0:
                wandb.log({"Validation Loss": loss.item()})
    return losses
# END: PARTIALLY COPIED FROM https://github.com/Shivanandroy/T5-Finetuning-PyTorch

def main(args):
    
    train_df = get_df_from_gcs_blob(args.gcs_train_blob, bucket=args.gcs_bucket)
    test_df = get_df_from_gcs_blob(args.gcs_test_blob, bucket=args.gcs_bucket)
    train_df = train_df[train_df['input'].map(str).map(len) < args.inp_max_len].reset_index(drop=True)

    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    TRAIN_NUM_WORKERS = args.train_num_workers
    TEST_NUM_WORKERS = args.test_num_workers
    MOD_SAVE_PATH = args.mod_save_path

    INP_MAX_LEN = max(train_df['input'].map(len).max(), test_df['input'].map(len).max())
    OUT_MAX_LEN = max(train_df['output'].map(len).max(), test_df['output'].map(len).max())

    MOD = 't5-small'
    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    FP16 = bool(args.fp16)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="recipe-t5",
        
        # track hyperparameters and run metadata
        config={
        "epochs": EPOCHS,
        "train_data_batch_size": TRAIN_BATCH_SIZE,
        "train_dataloader_num_workers": TRAIN_NUM_WORKERS,
        "test_data_batch_size": TEST_BATCH_SIZE,
        "test_dataloader_num_workers": TEST_NUM_WORKERS,
        "inp_max_len": INP_MAX_LEN,
        "out_max_len": OUT_MAX_LEN,
        "train_data_shape": train_df.shape,
        "test_data_shape": test_df.shape,
        "device": DEVICE,
        "lr": LR,
        "model": MOD,
        "fp16_enabled": FP16,
        "exp_name": args.exp_name
        }
    )

    tokenizer = T5Tokenizer.from_pretrained(MOD)
    #tokenizer.add_special_tokens({'additional_special_tokens': ['<ingredients>', '<calories>', '<title>', '<directions>']})

    train_dataset = T5Dataset(train_df['input'].values, train_df['output'].values, tokenizer, INP_MAX_LEN, OUT_MAX_LEN)
    test_dataset = T5Dataset(test_df['input'].values, test_df['output'].values, tokenizer, INP_MAX_LEN, OUT_MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=TEST_NUM_WORKERS)

    model = T5ForConditionalGeneration.from_pretrained(MOD).to(DEVICE)

    opt = torch.optim.Adam(params=model.parameters(), lr=LR)        
    
    for epoch in range(EPOCHS):
        print(f"Beginning training in epoch {epoch}...")
        start = perf_counter()
        train_losses = train(tokenizer, model, DEVICE, train_loader, opt, fp16=FP16)
        end = perf_counter()
        wandb.log({"Epoch Training Time (sec)": end - start})

        start = perf_counter()
        test_losses = test(tokenizer, model, DEVICE, test_loader, fp16=FP16)
        end = perf_counter()
        wandb.log({"Epoch Testing Time (sec)": end - start})

        epoch_running_train_loss = sum(train_losses)
        wandb.log({"Epoch Running Training Loss": epoch_running_train_loss})
        print(f"Epoch {epoch} Running Train Loss: {epoch_running_train_loss}")
        epoch_running_test_loss = sum(test_losses)
        wandb.log({"Epoch Running Testing Loss": epoch_running_test_loss})
        print(f"Epoch {epoch} Running Test Loss: {epoch_running_test_loss}")

        if epoch % 100 == 0:
            path = f'{MOD_SAVE_PATH}/{epoch}.pt'
            save_model(path, epoch, model, opt, epoch_running_train_loss)

    wandb.finish()
    print('Saving model and tokenizer!')
    model.save_pretrained(f'{MOD_SAVE_PATH}/final')
    tokenizer.save_pretrained(f'{MOD_SAVE_PATH}/final')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_bucket')
    parser.add_argument('--gcs_train_blob')
    parser.add_argument('--gcs_test_blob')
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--train_num_workers', type=int)
    parser.add_argument('--test_num_workers', type=int)
    parser.add_argument('--exp_name')
    parser.add_argument('--mod_save_path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--fp16', type=int)
    parser.add_argument('--inp_max_len', type=int, default=sys.maxsize)
    
    args = parser.parse_args()
    main(args)