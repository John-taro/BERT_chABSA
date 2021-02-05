from dataloader import get_chABSA_DataLoaders_and_TEXT
from bert import BertTokenizer
train_dl, val_dl, TEXT, dataloaders_dict= get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32)
#print(train_dl)

# 動作確認 検証データのデータセットで確認
batch = next(iter(train_dl))
print("Textの形状=", batch.Text[0].shape)
print("Labelの形状=", batch.Label.shape)
print(batch.Text)
print(batch.A_label)
print(batch.Label)

# ミニバッチの1文目を確認してみる
tokenizer_bert = BertTokenizer(vocab_file="./vocab/vocab.txt", do_lower_case=False)
text_minibatch_1 = (batch.Label).numpy()

# IDを単語に戻す
text = tokenizer_bert.convert_ids_to_tokens(text_minibatch_1)

print(text)
