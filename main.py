    
def main():
    train_on = 'obama'  # 'trump' or 'obama'
    val_size = 0.2
    max_len = 20
    embedding_size = 200
    hidden_size = 300
    batch_size = 64
    nb_epochs = 100
    max_grad_norm = 5
    teacher_forcing = 0.7

    # load data and create datasets
    # note that they use the same Vocab object so they will share the vocabulary
    # (in particular, for a given token both of them will return the same id)
    trump_tweets_filename = 'data/trump_tweets.txt'
    obama_tweets_filename = 'data/obama_white_house_tweets.txt'
    dataset_trump = TwitterFileArchiveDataset(trump_tweets_filename, max_len=max_len)
    dataset_obama = TwitterFileArchiveDataset(obama_tweets_filename, max_len=max_len, vocab=dataset_trump.vocab)

    dataset_trump.vocab.prune_vocab(min_count=3)

    if train_on == 'trump':
        dataset_train = dataset_trump
        dataset_val_ext = dataset_obama
    elif train_on == 'obama':
        dataset_train = dataset_obama
        dataset_val_ext = dataset_trump
    else:
        raise ValueError('`train_on` cannot be {} - use `trump` or `obama`'.format(train_on))

    val_len = int(len(dataset_train) * val_size)
    train_len = len(dataset_train) - val_len
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset_train, [train_len, val_len])

    # note that the the training and validation sets come from the same person,
    # whereas the val_ext set come from a different person

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    data_loader_val_ext = torch.utils.data.DataLoader(dataset_val_ext, batch_size=batch_size, shuffle=False)
    print('Training on: {}'.format(train_on))
    print('Train {}, val: {}, val ext: {}'.format(len(dataset_train), len(dataset_val), len(dataset_val_ext)))

    vocab_size = len(dataset_trump.vocab)
    model = NeuralLanguageModel(
        embedding_size, hidden_size, vocab_size,
        dataset_trump.vocab[dataset_trump.INIT_TOKEN], dataset_trump.vocab[dataset_trump.EOS_TOKEN],
        teacher_forcing
    )
    model = cuda(model)
    init_weights(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset_trump.vocab[dataset_trump.PAD_TOKEN])

    phases = ['train', 'val', 'val_ext']
    data_loaders = [data_loader_train, data_loader_val, data_loader_val_ext]

    losses_history = defaultdict(list)
    for epoch in range(nb_epochs):
        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = []
            for i, inputs in enumerate(data_loader):
                optimizer.zero_grad()

                inputs = variable(inputs)

                outputs = model(inputs)

                targets = inputs.view(-1)
                outputs = outputs.view(targets.size(0), -1)

                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(parameters, max_grad_norm)
                    optimizer.step()

                epoch_loss.append(float(loss))

            epoch_loss = np.mean(epoch_loss)
            print('Epoch {} {}\t\tloss {:.2f}'.format(epoch, phase, epoch_loss))
            losses_history[phase].append(epoch_loss)

            # decode something in the validation phase
            if phase == 'val_ext':
                possible_start_tokens = [
                    ['I','for' ],
                ]
                start_tokens = possible_start_tokens[np.random.randint(len(possible_start_tokens))]
                start_tokens = np.array([dataset_trump.vocab[t] for t in start_tokens])
                outputs = model.produce(start_tokens, max_len=20)
                outputs = outputs.cpu().numpy()

                produced_sequence = get_sequence_from_indices(outputs, dataset_trump.vocab.id2token)
                print('{}'.format(produced_sequence))

    print('Losses:')
    print('\t'.join(phases))
    losses = [losses_history[phase] for phase in phases]
    losses = list(zip(*losses))
    for losses_vals in losses:
        print('\t'.join('{:.2f}'.format(lv) for lv in losses_vals))
    PlotGraph = plotGraph.PlotGraph()
    PlotGraph.plot(nb_epochs = nb_epochs,loss=losses_history["train"],type = "training")
    PlotGraph.plot(nb_epochs = nb_epochs,loss=losses_history["val"],type = "validation")
    PlotGraph.plot(nb_epochs = nb_epochs,loss=losses_history["val_ext"],type = "validation extension")
 
    
if __name__ == '__main__':
    main()
