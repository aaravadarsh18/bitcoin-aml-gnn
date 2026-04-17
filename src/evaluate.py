from sklearn.metrics import classification_report

def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    
    print(classification_report(data.y.cpu(), pred.cpu()))