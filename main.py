# -*- coding: utf-8 -*-
"""
Main Script to Run the Classifier
"""
import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier import generate_data_sets, save_data, load_data
from feature_extraction import get_vocab, tokenize, get_label_idx
from nn import NN
from dataset import ArticleDataset


VOCAB_SIZE = 10000
MAX_ARTICLE_LEN = 100 
MAX_TITLE_LEN = 50
BATCH_SIZE = 8
EPOCHS = 30

if __name__ == "__main__":
    #train, val, test = generate_data_sets('./article_data.csv')
    #save_data(train, val, test)
    print('Loading dataset')
    train, val, test = load_data()
    
    # Generate the vocabulary based on the top TF-IDF weighted words
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    vectorizer, data = get_vocab(train['title'], train['publication'], length=VOCAB_SIZE, vocab_type='tf-idf')
    
    # Removing words not in pretrained word2vec from vocabulary
    print('Original Vocabulary Length: ', len(vectorizer.vocabulary_))
    out_of_vocab = []
    for word in vectorizer.vocabulary_.keys():
        if word not in word2vec.vocab.keys():
            out_of_vocab.append(word)
    
    for word in out_of_vocab:
        vectorizer.vocabulary_.pop(word)
    
    print('Word2Vec Vocabulary Length: ', len(vectorizer.vocabulary_))
    tokenizer = vectorizer.build_tokenizer()

    publications = train['publication'].unique().tolist()

    # Creating DataLoader to handle tokenization and batch creation
    trainset = ArticleDataset(train, tokenizer, vectorizer.vocabulary_.keys(), 1, 
                              word2vec, publications, MAX_ARTICLE_LEN, MAX_TITLE_LEN)

    valset = ArticleDataset(val, tokenizer, vectorizer.vocabulary_.keys(), 1,
                            word2vec, publications, MAX_ARTICLE_LEN, MAX_TITLE_LEN)
    
    testset = ArticleDataset(test, tokenizer, vectorizer.vocabulary_.keys(), 1,
                             word2vec, publications, MAX_ARTICLE_LEN, MAX_TITLE_LEN)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    # Intialize a NN
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    model = NN(300*(MAX_ARTICLE_LEN + MAX_TITLE_LEN), [1000], device)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    print('Training')
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, data in enumerate(trainloader):            
            batch, target = data[0].to(device), data[1].to(device)
            
            model.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_loss += loss.item()
            del output
            if i % (len(trainloader)/10) == 0:
                print('{}/{}, Loss: {:.4f}'.format(i, len(trainloader), total_loss))
        
        print("\t({}/{}): Total Loss: {:.4f}, Avg Loss: {:.4f}".format(epoch, EPOCHS, total_loss, total_loss/len(trainloader)))
    
    torch.save(model.state_dict(), './nn_1000.pt')
    
    # Finished Training
    with torch.no_grad():
        model.to(device)
        val_correct = 0
        val_loss = 0
        for i, data in enumerate(valloader):
            batch, target = data[0].to(device), data[1].to(device)
            model.zero_grad()
            output = model(batch)
            loss = criterion(output.squeeze(), target)
            val_correct += np.sum(torch.argmax(output.cpu()).numpy() == target.cpu().numpy())
            val_loss += loss.cpu().detach().numpy()
            del loss, output
            if i % (len(valloader)/10) == 0:
                print('{}/{}, Loss: {:.4f}, Correct: {}/{}'.format(i, len(valloader), val_loss, val_correct, i*BATCH_SIZE))
                
        val_accuracy = val_correct / (len(valloader) * valloader.batch_size)
        
        train_correct = 0
        train_loss = 0
        for i, data in enumerate(trainloader):
            batch, target = data[0].to(device), data[1].to(device)
            model.zero_grad()
            output = model(batch)
            loss = criterion(output.squeeze(), target)
            train_correct += np.sum(torch.argmax(output).cpu().detach().numpy() == target.cpu().detach().numpy())
            train_loss += loss.cpu().detach().numpy()
            del loss, output
            if i % (len(trainloader)/10) == 0:
                print('{}/{}, Loss: {:.4f}, Correct: {}/{}'.format(i, len(trainloader), train_loss, train_correct, i*BATCH_SIZE))
            
        train_accuracy = train_correct / (len(trainloader) * trainloader.batch_size)
        
    predicted = []
    for i in range(len(train_tokenized)):
        output = model(train_tokenized[i])
        predicted.append(torch.argmax(output).detach().numpy())
    
    predicted = np.array(predicted)
    np.sum(predicted == train_labels)