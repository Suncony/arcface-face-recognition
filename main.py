from data_preparation import *
from model import *
from train import *
from utils import *

def main():
    print("Preparing datasets...")
    train_loader, val_loader, num_classes = prepare_data()
    
    models_config = {
        'ArcFace': {
            'model': ArcFaceModel(num_classes),
            'criterion': SmoothCrossEntropyLoss(smoothing=0.1),
            'lr': 0.0003,
            'weight_decay': 0.01
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}")
        model = config['model'].to(device)
        criterion = config['criterion']
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=22,
            min_lr=1e-6,
            verbose=True
        )
        
        history, best_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=120,
            model_name=model_name,
            patience=30
        )
        
        results[model_name] = {
            'history': history,
            'best_acc': best_acc
        }
        
        print(f"{model_name}: Best Validation Accuracy = {best_acc:.4f}")
        
    print("\nModel Comparison Results:")
    for model_name, result in results.items():
        print(f"{model_name}: Best Validation Accuracy = {result['best_acc']:.4f}")
    
    # Plot training progress for all models
    plot_training_progress(results)

if __name__ == "__main__":
    main()