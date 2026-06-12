import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name, patience=30):
    scaler = GradScaler()
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stop_count = 0
    
    # Add weight decay
    if isinstance(optimizer, optim.AdamW):
        optimizer.param_groups[0]['weight_decay'] = 0.01
    
    # Learning rate warmup
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                 start_factor=0.1,
                                                 end_factor=1.0,
                                                 total_iters=warmup_epochs * len(train_loader))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixup augmentation
            if epoch > warmup_epochs:
                lam = np.random.beta(0.5, 0.5)
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                inputs = mixed_inputs
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                if epoch > warmup_epochs:
                    loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Calculate training accuracy and loss
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        
        # Update learning rate scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'best_{model_name}.pth')
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
            
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
    return history, best_acc


def plot_training_progress(results):
    num_models = len(results)
    plt.figure(figsize=(15, num_models * 5))
    
    for i, (model_name, result) in enumerate(results.items(), 1):
        train_losses = result['history']['train_loss']
        val_losses = result['history']['val_loss']
        train_accs = result['history']['train_acc']
        val_accs = result['history']['val_acc']
        
        # Loss Plot
        plt.subplot(num_models, 2, 2 * i - 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_name} - Loss')

        # Accuracy Plot
        plt.subplot(num_models, 2, 2 * i)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'{model_name} - Accuracy')
    
    plt.tight_layout()
    plt.show()