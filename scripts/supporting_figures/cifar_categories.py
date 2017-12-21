import nengo_extras.data

train, test, label_names = nengo_extras.data.load_cifar100(label_names=True)
for s in label_names:
    print(s)
