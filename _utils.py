import torch 

def evaluate_manually(model,data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            label = torch.argmax(out, dim=1) 
            correct += torch.sum(label == data.y)
            total += data.y.shape[0]
    return (correct/total).clone().detach().cpu().numpy()
