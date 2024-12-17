import torch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    # Set the model to training mode
    model.train()
    
    # Iterate through the specified number of epochs
    for epoch in range(num_epochs):
        epoch_loss = 0  # Initialize epoch loss to 0
        
        # Iterate through batches in the training data
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass: compute the model outputs
            outputs = model(images)
            
            # Compute the loss between predictions and ground truth
            loss = criterion(outputs, masks)
            
            # Zero the gradients before backward pass to avoid accumulation
            optimizer.zero_grad()
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters based on computed gradients
            optimizer.step()
            
            # Accumulate batch loss
            epoch_loss += loss.item()
        
        # Print average loss for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
