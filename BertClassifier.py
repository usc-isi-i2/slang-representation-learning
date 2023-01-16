from torch import nn
from transformers import BertModel, RobertaForSequenceClassification
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report


class BertClassifier(nn.Module):

    def __init__(self, bert_model, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
#         print(pooled_output.shape)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model,tokenizer, train_data, val_data, learning_rate=1e-6, epochs=5):
    train, val = TweetDataset(train_data, tokenizer), TweetDataset(val_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, tokenizer, test_data):
    test = TweetDataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    CM = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()

            total_acc_test += acc
            CM += confusion_matrix(test_label.cpu(), output.argmax(dim=1).cpu(), labels=[0, 1])
    tn = CM[0][0]
    tp = CM[1][1]
    fp = CM[0][1]
    fn = CM[1][0]
    cm_acc = np.sum(np.diag(CM) / np.sum(CM))
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)

    print('\nTestset Accuracy(mean): %f %%' % (100 * cm_acc))
    print()
    print('Confusion Matirx : ')
    print(CM)
    print('- Sensitivity : ', (tp / (tp + fn)) * 100)
    print('- Specificity : ', (tn / (tn + fp)) * 100)
    print('- Precision: ', (tp / (tp + fp)) * 100)
    print('- NPV: ', (tn / (tn + fn)) * 100)
    print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
    print()

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def evaluate_semeval(model, tokenizer, test_data):
    test = TweetDataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    y_pred = []
    y_true = []

    total_acc_test = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            
            total_acc_test += acc
            y_pred.extend(output.argmax(dim=1).cpu().detach().numpy())
            y_true.extend(test_label.cpu().detach().numpy())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4))
  
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
