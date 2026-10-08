#%%
import os, sys
import numpy as np
import pandas as pd
import openpyxl
import torch
from torch import nn
from matplotlib import pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
#%% md
## Define Experiment Parameters
#%%
is_local = False
model_name = 'unet'

if is_local:
    sys.path.append(r'C:/Users/w37262do/Documents/git/IR_preprocessing')
else:
    sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing')
from src.utils import ir_core_dataset_torch, statistics_from_loader, adjust_learning_rate
from src.models import UNet
#%%
# Experiment
seed = int(sys.argv[1]) if not is_local else 10447201
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
gen = torch.Generator()
gen.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Data
wavenumbers = np.load(r"D:\datasets\qcl2026\wavenumbers.npy" if is_local else './wavenumbers.npy')
num_wavenumbers = len(wavenumbers)
input_image_dims = np.array([num_wavenumbers, 360, 360])
annotation_class_colors = np.array([[0,255,0],[128,0,128],[255,0,255],[0,0,255],[255,165,0],[255,0,0], [255,255,0]])
annotation_class_names = np.array(['epithelium_n','stroma_n','epithelium_c','stroma_c','corpora_amylacea','blood','immune_inf'])
n_classes = len(annotation_class_names)
standardise_data = bool(int(sys.argv[2])) if not is_local else 0

# Data loading
test_set_fraction = 0.2
val_set_fraction= 0.2
use_augmentation = True
num_workers = 0 if is_local else 6
prefetch_factor = None if is_local else 2
use_augmentation = True

# Training
batch_size = 8
initial_lr = 1e-4
l2 = 5e-2
max_epochs = 100
warmup_epochs = 5
min_lr = 1e-7
save_best_model = True
#%% md
## Import Data Sheets
#%%
master_nocl = pd.read_excel(r'D:/datasets/qcl2026/master_sheet.xlsx' if is_local else './qcl2026_processed/master_sheet.xlsx')
master_aus = pd.read_excel(r'D:/datasets/aus_qcl2026/master_sheet.xlsx' if is_local else './aus_qcl2026_processed/master_sheet.xlsx')

# Add offset to patient ID in AUS dataset so no false associations
master_nocl['patient_id'] = (master_nocl['patient_id'].to_numpy() + 1000000).astype(str)
#%%
# Load NOCL
pid_nocl = master_nocl['patient_id'].to_numpy()
nocl_base = r'D:/datasets/qcl2026_raw/' if is_local else r'./qcl2026_processed/'
h5fp_nocl = np.array([f"{nocl_base}spectral/{i.split('/')[-1].replace('.h5','.pt')}" for i in master_nocl['hdf5_filepath'].to_numpy()])
annofp_nocl = np.array([f"{nocl_base}annotation/{i.split('/')[-1].replace('.h5','.png')}" for i in master_nocl['hdf5_filepath'].to_numpy()])
maskfp_nocl = np.array([f"{nocl_base}mask/{i.split('/')[-1].replace('.h5','.png')}" for i in master_nocl['hdf5_filepath'].to_numpy()])

# Load AUS
aus_base = r'F:/datasets/aus_qcl2026_raw/' if is_local else r'./aus_qcl2026_processed/'
pid_aus = master_aus['patient_id'].to_numpy()
h5fp_aus = np.array([f"{aus_base}spectral/{i.split('/')[-1].replace('.h5','.pt')}" for i in master_aus['hdf5_filepath'].to_numpy()])
annofp_aus = np.array([f"{aus_base}annotation/{i.split('/')[-1].replace('.h5','.png')}" for i in master_aus['hdf5_filepath'].to_numpy()])
maskfp_aus = np.array([f"{aus_base}mask/{i.split('/')[-1].replace('.h5','.png')}" for i in master_aus['hdf5_filepath'].to_numpy()])
#%% md
## Define Datasets, Dataloaders
#%%
unique_pids_nocl = np.unique(pid_nocl)
pids_trainval, pids_test, _, _ = train_test_split(
    unique_pids_nocl, np.zeros_like(unique_pids_nocl), test_size=test_set_fraction, random_state=seed)
pids_train, pids_val, _, _ = train_test_split(
    pids_trainval, np.zeros_like(pids_trainval), test_size=(val_set_fraction/(1-test_set_fraction)), random_state=seed)
where_train = np.where(np.isin(pid_nocl,pids_train))
where_val = np.where(np.isin(pid_nocl,pids_val))
where_test = np.where(np.isin(pid_nocl,pids_test))

print(f"Cores per data split:\n\tTRAIN: {len(where_train[0])}\n\tVAL: {len(where_val[0])}\n\tTEST: {len(where_test[0])}")
print(f"Independent cohort: {len(pid_aus)}")
#%%
dataset_train = ir_core_dataset_torch(
    h5fp_nocl[where_train], maskfp_nocl[where_train], annofp_nocl[where_train],
    annotation_class_names, annotation_class_colors,
    augment=use_augmentation,generator=gen
)
dataset_val = ir_core_dataset_torch(
    h5fp_nocl[where_val], maskfp_nocl[where_val], annofp_nocl[where_val],
    annotation_class_names, annotation_class_colors,
    augment=False,generator=gen
)
dataset_test = ir_core_dataset_torch(
    h5fp_nocl[where_test], maskfp_nocl[where_test], annofp_nocl[where_test],
    annotation_class_names, annotation_class_colors,
    augment=False,generator=gen
)
dataset_aus = ir_core_dataset_torch(
    h5fp_aus, maskfp_aus, annofp_aus,
    annotation_class_names, annotation_class_colors,
    augment=False,generator=gen
)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,drop_last=True, generator=gen, num_workers=num_workers, prefetch_factor=prefetch_factor)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,drop_last=False, generator=gen, num_workers=num_workers, prefetch_factor=prefetch_factor)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=False,drop_last=False, generator=gen, num_workers=num_workers, prefetch_factor=prefetch_factor)
aus_loader = torch.utils.data.DataLoader(dataset_aus, batch_size=batch_size,shuffle=False,drop_last=False, generator=gen, num_workers=num_workers, prefetch_factor=prefetch_factor)
print(f"loader sizes:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}\n\ttest: {len(test_loader)}")
print(f"Independent cohort: {len(aus_loader)}")
#%%
if is_local: # shortcut for quick error-check
    train_mean = torch.from_numpy(np.load(r'D:/datasets/qcl2026/mean_spectrum.npy')).reshape(1,num_wavenumbers,1,1).to(device)
    train_std = torch.from_numpy(np.load(r'D:/datasets/qcl2026/std_spectrum.npy')).reshape(1,num_wavenumbers,1,1).to(device)
else:
    train_mean, train_std, train_px = statistics_from_loader(train_loader,num_wavenumbers)
    train_mean = train_mean.reshape(1,num_wavenumbers,1,1).to(device)
    train_std = train_std.reshape(1,num_wavenumbers,1,1).to(device)

train_std[torch.where(train_std==0)] = 1 # some transforms set every wavenumber to 0. this gives div by zero error for std

def norm(core, mask):
    if standardise_data:
        core -= train_mean
        core /= train_std
    return core
#%% md
## Define Model
#%%
model = UNet(num_wavenumbers, num_wavenumbers, n_classes, reduce_method='none')
model = model.to(device)
model.eval()
#%% md
## Train Loop
#%%
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=initial_lr,
    weight_decay=l2,
    betas=(0.9, 0.999)
)
loss_fn = nn.CrossEntropyLoss(reduction='none')

training_losses,validation_losses = [],[]
training_accs,validation_accs = [],[]
training_f1ms,validation_f1ms = [],[]
training_f1s,validation_f1s = [],[]
learning_rates = []
best_val_f1 = 0
best_val_epoch = 0
#%%
for epoch in range(max_epochs):
    curr_lr = adjust_learning_rate(optimiser, epoch, max_epochs, warmup_epochs, initial_lr, min_lr)
    learning_rates.append(curr_lr)
    print(f"\n ON EPOCH {epoch + 1}/{max_epochs} | LR = {curr_lr}")

    # reset running metrics
    running_loss_train, running_loss_val = 0, 0
    train_preds, train_targets = [], []
    val_preds, val_targets = [], []

    # Train loop
    model.train()
    for batch_idx, (data, annot, mask, has_annotations) in enumerate(train_loader):

        # Put data and label on device
        data = data.to(device); mask = mask.to(device); annot = annot.to(device);
        has_annotations = has_annotations.to(device)
        data = norm(data, mask)

        # Push data through model
        optimiser.zero_grad()
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel with annotations
        loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel
        loss.backward()  # backprop
        optimiser.step()

        # Calculate metrics
        running_loss_train += loss.cpu().item()
        targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
        preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(targets.detach().cpu().numpy())

    # Validate loop
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, annot, mask, has_annotations) in enumerate(val_loader):

            # Put data and label on device
            data = data.to(device); annot = annot.to(device); mask = mask.to(device)
            has_annotations = has_annotations.to(device)
            data = norm(data, mask)

            # Push data through model
            out = model(data)

            # Calculate loss
            loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel
            loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel

            # Calculate metrics
            running_loss_val += loss.cpu().item()
            targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
            preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
            val_preds.extend(preds.detach().cpu().numpy())
            val_targets.extend(targets.detach().cpu().numpy())

    # calculate epoch metrics for train set
    train_acc = accuracy_score(train_targets, train_preds); training_accs.append(train_acc)
    train_f1m = f1_score(train_targets, train_preds, average='macro'); training_f1ms.append(train_f1m)
    train_f1 = f1_score(train_targets, train_preds, average=None); training_f1s.append(train_f1)
    train_loss = running_loss_train / (len(dataset_train));
    training_losses.append(train_loss)

    # calculate epoch metrics for val set
    val_acc = accuracy_score(val_targets, val_preds); validation_accs.append(val_acc)
    val_f1m = f1_score(val_targets, val_preds, average='macro'); validation_f1ms.append(val_f1m)
    val_f1 = f1_score(val_targets, val_preds, average=None); validation_f1s.append(val_f1)
    val_loss = running_loss_val / (len(dataset_val));
    validation_losses.append(val_loss)

    # Update
    print(f"TRAIN --- | Loss: {train_loss:.4} | OA: {train_acc:.4} | f1: {train_f1m:.4}")
    print(f"VAL ----- | Loss: {val_loss:.4} | OA: {val_acc:.4} | f1: {val_f1m:.4}")

    # If performance on validation set best so far, save model
    if val_f1m > best_val_f1:
        best_val_f1 = val_f1m
        best_val_epoch = epoch
        torch.save(
            model.state_dict(),
            rf'./{model_name}_weights_{seed}.pt' if not is_local else rf'../model_weights/{model_name}_weights_{seed}.pt'
        )
#%%
if save_best_model: # if saving best model, we want to load the best model now for evaluation
    model.load_state_dict(
        torch.load(
            rf'./{model_name}_weights_{seed}.pt' if not is_local else rf'../model_weights/{model_name}_weights_{seed}.pt',
            weights_only=True)
    )
model.eval()
#%% md
## Test Loop
#%%
# Test - NOCL
running_loss_test = 0
test_preds, test_targets = [], []
model.eval()
with torch.no_grad():
    for batch_idx, (data, annot, mask, has_annotations) in enumerate(test_loader):
        # Put data and label on device
        data = data.to(device); mask = mask.to(device); annot = annot.to(device);
        has_annotations = has_annotations.to(device)
        data = norm(data, mask)

        # Push data through model
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel
        loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel

        # Calculate metrics
        running_loss_test += loss.cpu().item()
        targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
        preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
        test_preds.extend(preds.detach().cpu().numpy())
        test_targets.extend(targets.detach().cpu().numpy())

# calculate test set metrics
test_acc = accuracy_score(test_targets, test_preds)
test_f1m = f1_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average=None)
test_loss = running_loss_test / len(dataset_test)
#%%
# Test - NOCL
aus_running_loss_test = 0
aus_test_preds, aus_test_targets = [], []
model.eval()
with torch.no_grad():
    for batch_idx, (data, annot, mask, has_annotations) in enumerate(aus_loader):
        # Put data and label on device
        data = data.to(device); mask = mask.to(device); annot = annot.to(device);
        has_annotations = has_annotations.to(device)
        data = norm(data, mask)

        # Push data through model
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel
        loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel

        # Calculate metrics
        aus_running_loss_test += loss.cpu().item()
        targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
        preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
        aus_test_preds.extend(preds.detach().cpu().numpy())
        aus_test_targets.extend(targets.detach().cpu().numpy())

# calculate test set metrics
aus_test_acc = accuracy_score(aus_test_targets, aus_test_preds)
aus_test_f1m = f1_score(aus_test_targets, aus_test_preds, average='macro')
aus_test_f1 = f1_score(aus_test_targets, aus_test_preds, average=None)
aus_test_loss = aus_running_loss_test / len(dataset_aus)
#%%
print(f"TEST (NOCL) ---- | Loss: {test_loss:.4} | OA: {test_acc:.4} | f1: {test_f1m:.4}")
for cls_idx, f1 in enumerate(test_f1):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4}")

print(f"TEST (AUS) ----- | Loss: {aus_test_loss:.4} | OA: {aus_test_acc:.4} | f1: {aus_test_f1m:.4}")
for cls_idx, f1 in enumerate(aus_test_f1):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4}")
#%% md
## Evaluate Training Process
#%%
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(np.arange(0,len(learning_rates)),learning_rates,color='red',label="lr")
ax.legend(); ax.grid(); ax.set_title("Learning Rate / Epoch")
if not is_local:
    plt.savefig(f'./lr_curves_{model_name}_{seed}.png')
    plt.close(fig)
#%%
# Plot overall performance curves per-epoch
fig,ax = plt.subplots(1,3,figsize=(16,5))
ax[0].plot(np.arange(1,len(training_losses)+1),np.array(training_losses),color='cornflowerblue',label="train")
ax[0].plot(np.arange(1,len(validation_losses)+1),np.array(validation_losses),color='orange',label="validation")
ax[0].scatter(len(validation_losses),test_loss,color='green',label="test",marker="x")
ax[0].scatter(len(validation_losses),aus_test_loss,color='crimson',label="test (aus)",marker="*")
ax[0].set_title("loss curves"); ax[0].legend()

ax[1].plot(np.arange(1,len(training_accs)+1),np.array(training_accs),color='cornflowerblue',label="train")
ax[1].plot(np.arange(1,len(validation_accs)+1),np.array(validation_accs),color='orange',label="validation")
ax[1].scatter(len(validation_losses),test_acc,color='green',label="test",marker="x")
ax[1].scatter(len(validation_losses),aus_test_acc,color='crimson',label="test (aus)",marker="*")
ax[1].set_title("accuracy"); ax[1].legend(); ax[1].set_ylim(ymin=0,ymax=1)

ax[2].plot(np.arange(1,len(training_f1ms)+1),np.array(training_f1ms),color='cornflowerblue',label="train")
ax[2].plot(np.arange(1,len(validation_f1ms)+1),np.array(validation_f1ms),color='orange',label="validation")
ax[2].scatter(len(validation_losses),test_f1m,color='green',label="test",marker="x")
ax[2].scatter(len(validation_losses),aus_test_f1m,color='crimson',label="test (aus)",marker="*")
ax[2].set_title("macro f1"); ax[2].legend(); ax[2].set_ylim(ymin=0,ymax=1)

ax[0].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
ax[1].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
ax[2].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)

fig.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curves_{model_name}_{seed}.png')
    plt.close(fig)
#%%
# Plot per-class f1 score curves
fig,ax = plt.subplots(2,4,figsize=(16,4)); ax = ax.flatten()
for cls in range(7):
    ax[cls].plot(np.arange(1,len(training_f1s)+1),[i[cls] for i in training_f1s], color='black', label="train")
    ax[cls].plot(np.arange(1,len(validation_f1s)+1),[i[cls] for i in validation_f1s], color=annotation_class_colors[cls]/255, label="val")
    ax[cls].set_title(f"{annotation_class_names[cls]}")
    ax[cls].legend()
    ax[cls].scatter(len(validation_losses),test_f1[cls],color='green',label="test",marker="x")
    ax[cls].scatter(len(validation_losses),aus_test_f1[cls],color='crimson',label="test (aus)",marker="*")
    ax[cls].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
    ax[cls].set_ylim(ymin=0,ymax=1)
ax[7].plot(np.arange(1,len(training_f1ms)+1),np.array(training_f1ms),color='cornflowerblue',label="train")
ax[7].plot(np.arange(1,len(validation_f1ms)+1),np.array(validation_f1ms),color='orange',label="validation")
ax[7].scatter(len(validation_losses),test_f1m,color='green',label="test",marker="x")
ax[7].scatter(len(validation_losses),aus_test_f1m,color='crimson',label="test (aus)",marker="*")
ax[7].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3); ax[7].set_title("macro f1"); ax[7].legend()
ax[7].set_ylim(ymin=0,ymax=1)
fig.suptitle("Class-specific F1 scores")
plt.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curves_byclass_{model_name}_{seed}.png')
    plt.close(fig)
#%%
# Normalised such that each 'actual label' column sums to 1. Each entry is thus the fraction of actual label predicted correctly
cm = confusion_matrix(test_targets, test_preds, labels=np.arange(0,n_classes))
cm_norm = confusion_matrix(test_targets, test_preds, labels=np.arange(0,n_classes), normalize='true')
roffset = 0.07; coffset = -0.32

fig,ax = plt.subplots(1,2, figsize=(11,5))
ax[0].matshow(cm,cmap='magma',aspect=1); ax[1].matshow(cm_norm,cmap='magma',aspect=1)
for x in range(2):
    ax[x].set_xticks(np.arange(0, n_classes),labels=annotation_class_names,rotation=25)
    ax[x].set_yticks(np.arange(0, n_classes),labels=annotation_class_names,rotation=25)
    ax[x].set_xlabel("predicted label"); ax[x].xaxis.set_label_position("top"); ax[x].set_ylabel("actual label")
for r in range(n_classes):
    for c in range(n_classes):
        ax[0].text(y=r+roffset,x=c+coffset,s=f'{cm[r,c]}',fontdict={'fontsize':10,'color':"k" if cm[r,c] > cm.max()/2 else "w"})
        ax[1].text(y=r+roffset,x=c+coffset,s=f'{cm_norm[r,c]:.3f}',fontdict={'fontsize':10,'color':"k" if cm_norm[r,c] > 0.5 else "w"})
fig.supxlabel(f"{' '*35}confusion matrix, total pixels{' '*50}confusion matrix, normalised along actual"); fig.tight_layout()
if not is_local:
    plt.savefig(f'./confusion_{model_name}_{seed}.png')
    plt.close(fig)
#%% md
## Plot Segmentation
#%%
for loader,dset,name in zip([test_loader,aus_loader],[dataset_test,dataset_aus],["nocl","aus"]):
    ftir, annotations, tissue_mask, has_annotations = next(iter(loader))
    model.eval()
    with torch.no_grad():
        out = model(norm(ftir[:4].to(device),mask[:4].to(device)))
        pred = out.argmax(dim=1).cpu().numpy()
        predcol = annotation_class_colors[pred] / 255.0

    fig,ax = plt.subplots(2,4,figsize=(16,8))
    for x in range(4):
        ax[0,x].matshow(cv2.imread(dset.annotation_filepaths[x])[:,:,::-1])
        ax[1,x].matshow(predcol[x] * tissue_mask[x].squeeze(0).unsqueeze(-1).numpy())
        ax[0,x].set_axis_off(); ax[1,x].set_axis_off()
        if x == 0:
            ax[0,x].text(x=5,y=15,s=f'annotation',fontdict={'fontweight':'bold','fontsize':15,'color':'cyan'})
            ax[1,x].text(x=5,y=15,s=f'prediction',fontdict={'fontweight':'bold','fontsize':15,'color':'cyan'})
    fig.tight_layout()
    if not is_local:
        plt.savefig(f'./pred_{name}_{model_name}_{seed}.png')
        plt.close(fig)
#%% md
## Save Model and Results
#%%
# Save model
torch.save(
    model.state_dict(),
    rf'./{model_name}_weights_{seed}.pt' if not is_local else rf'../model_weights/{model_name}_weights_{seed}.pt'
)

# Save mean and std of data
torch.save(
    train_mean.cpu(),
    rf'./{model_name}_train_mean_{seed}.pt' if not is_local else rf'../model_weights/{model_name}_train_mean_{seed}.pt'
)
torch.save(
    train_std.cpu(),
    rf'./{model_name}_train_std_{seed}.pt' if not is_local else rf'../model_weights/{model_name}_train_std_{seed}.pt'
)

# save results
if not is_local:
    # NOCL
    if os.path.isfile(f'results_NOCL_{model_name}.txt'):
        f = open(f'results_NOCL_{model_name}.txt', 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [x + ', \n' for x in ['seed', *annotation_class_names, 'overall_acc', 'macro_f1']]

    # Process files
    lines[0] = lines[0].replace('\n', str(seed) + ', \n')
    for cls in range(n_classes):
        lines[cls + 1] = lines[cls + 1].replace('\n', str(test_f1[cls]) + ', \n')
    lines[n_classes + 1] = lines[n_classes + 1].replace('\n', str(test_acc) + ', \n')
    lines[n_classes + 2] = lines[n_classes + 2].replace('\n', str(test_f1m) + ', \n')

    f = open(f'results_NOCL_{model_name}.txt', 'w')
    f.write(''.join(lines))
    f.close()

    # AUS
    if os.path.isfile(f'results_AUS_{model_name}.txt'):
        f = open(f'results_AUS_{model_name}.txt', 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [x + ', \n' for x in ['seed', *annotation_class_names, 'overall_acc', 'macro_f1']]

    # Process files
    lines[0] = lines[0].replace('\n', str(seed) + ', \n')
    for cls in range(n_classes):
        lines[cls + 1] = lines[cls + 1].replace('\n', str(aus_test_f1[cls]) + ', \n')
    lines[n_classes + 1] = lines[n_classes + 1].replace('\n', str(aus_test_acc) + ', \n')
    lines[n_classes + 2] = lines[n_classes + 2].replace('\n', str(aus_test_f1m) + ', \n')

    f = open(f'results_AUS_{model_name}.txt', 'w')
    f.write(''.join(lines))
    f.close()