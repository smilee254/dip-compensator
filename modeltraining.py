import matplotlib.pyplot as plt
import seaborn as sns
def train(X_train, y_train, model):
    return model.fit(X_train, y_train)

#test model perfomance
def perfo(pred, y_test, model):
    acc = accuracy_score(y_test, pred)*100
    prec = precision_score(y_test, pred, average='weighted')*100
    rec = recall_score(y_test, pred, average='weighted')*100
    f1 = f1_score(y_test, pred, average='weighted')*100

    print(f'{model} performances')
    print(f'Accuracy:{acc:.2f}%')
    print(f'precision:{prec:.2f}%')
    print(f'recall:{rec:.2f}%')
    print(f'F1_score:{f1:.2f}%')
    print()

    c = cm(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(7, 5))
    sns.heatmap(c, annot=True, fmt="d", cmap="Blues", xticklabels=["+ve", "-ve"], yticklabels=["+ve", "-ve"])
    plt.title(f"confusion Matrix-{model}")
    plt.xlabel('predicted')
    plt.ylabel('True')
    plt.show()

trianed = dict()
for m in models:
    model = models.get(m)

    t = train(X_train=X_train, y_train=y_train, model=model)
    trained.update({m: t})

    pred = t.predict(X_test)

    p = perfo(pred=pred, y_test=y_test, model=m)