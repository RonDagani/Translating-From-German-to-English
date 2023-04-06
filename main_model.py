import numpy as np
import spacy
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    Adafactor, BertTokenizer, BertModel
import project_evaluate
from datasets import Dataset
from transformers import T5Tokenizer
from preprocess import LANG_Dataset
from sklearn.metrics.pairwise import cosine_similarity


def hug_face_dataset(dataset):
    """
    The function get the sentences in german and english and make a hugging face out of them
    :param dataset: the data which will be turned into a dataset
    :return: hugging face dataset out of the data
    """
    new_dict = {'ger': dataset[1], 'eng': dataset[0]}
    return Dataset.from_dict(new_dict)


def process_data(example):
    """
    The function adds a prefix needed for the t5 algorithm and tokenized
    :param example: a record from the hugging face dataset 
    :return: the example after tokenization
    """
    # adding the prefix
    prefix = "translate German to English: "
    inputs = [prefix + sn for sn in example['ger']]
    targets = [sn for sn in example['eng']]
    # tokenization
    model_inputs = tokenizer_t5(inputs, max_length=210, truncation=True)
    with tokenizer_t5.as_target_tokenizer():
        labels = tokenizer_t5(targets, max_length=210, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics_new(prediction_labels):
    """
    The function calculates the blue score of the predictions according to the real translations.
    :param prediction_labels: the predictions of the model
    :return: the outputs of the metrics calculated
    """
    predictions, labels = prediction_labels
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    decoded_predictions = tokenizer_t5.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer_t5.pad_token_id)
    decoded_labels = tokenizer_t5.batch_decode(labels, skip_special_tokens=True)
    # calculate according to the given compute metrics function
    result = project_evaluate.compute_metrics(decoded_predictions, decoded_labels)
    result = {"bleu": result}
    # write results into a file
    with open('results.txt', 'a') as f:
        f.write(str(result))
    result_list.append(result)
    print(result_list)
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer_t5.pad_token_id) for pred in predictions])
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)
    return result


def modifier_matcher(real_modfiers, our_modfiers):
    """
    given real and found modfiers replace the found ones with most similar real ones
    :param real_modfiers: the modifiers the english sentences received
    :param our_modfiers: the modifiers we found
    :return: modfiers to be replaced
    """
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')

    real_modfiers_bert = torch.stack(
        [model_bert(torch.tensor(tokenizer_bert.encode(word, return_tensors="pt"))).pooler_output for word in
         real_modfiers]).squeeze(1)
    our_modfiers_bert = torch.stack(
        [model_bert(torch.tensor(tokenizer_bert.encode(str(word), return_tensors="pt"))).pooler_output for word in
         our_modfiers]).squeeze(1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_matrix = cosine_similarity(our_modfiers_bert.cpu().detach().numpy(),
                                      real_modfiers_bert.cpu().detach().numpy())
    replace_mod_max = []
    for i, mod in enumerate(our_modfiers):
        max_our_mod = np.argmax(cosine_matrix[i])
        replace_mod_max.append([real_modfiers[max_our_mod], mod])
    return replace_mod_max


def replace_root_mods(all_paragraphs, no_trans_set, en_nlp):
    """

    returns all the roots and modifiers of the paragraphs given
    :param all_paragraphs: paragraphs to find roots and modifiers for
    :param no_trans_set: the set which the modfiers and roots are taken from
    :param en_nlp: spacy model
    :return: the paragraphs with replaced roots and modifiers according to the real ones
    """
    all_roots, all_mods = [], []
    total_replace_mod = []
    modifiers = no_trans_set.modifiers_english
    roots = no_trans_set.roots_english
    second_phase_word = []
    prag_words = []

    for i in all_paragraphs:
        for word in i.split(' '):
            prag_words.append(word)
        second_phase_word.append(prag_words)
        prag_words = []

    for p, paragraph in enumerate(all_paragraphs):
        cur_roots, cur_mods = root_and_modefiers(paragraph, en_nlp)
        for s in range(len(cur_roots)):
            second_phase_word[p][cur_roots[s][1]] = cur_roots[s][0]
            if len(cur_mods[s]) != 0:
                for mod_inx in range(len(cur_mods[s])):
                    cur_replacement = modifier_matcher(modifiers[p][s], cur_mods[s])
                    second_phase_word[p][cur_mods[s][mod_inx][2]] = cur_replacement[mod_inx][0]
        all_roots.append(cur_roots)
        all_mods.append(cur_mods)

    final_phase_word = []
    for i in second_phase_word:
        final_phase_word.append(' '.join(i))
    return final_phase_word


def root_and_modefiers(paragraph, en_nlp):
    """

    :param paragraph: a paragraph which the function needs to find its roots and modefiers
    :param en_nlp: spacy model
    :return: the roots and the modefiers of the paragraph
    """
    roots = []
    mods = []
    doc = en_nlp(paragraph)
    list_paragraph = paragraph.split(" ")
    dep = []
    for sentence in doc.sents:
        roots.append([str(sentence.root), list_paragraph.index(str(sentence.root))])
        for token in list(sentence.root.children):
            # if "mod" in token.dep_ and token.dep_!="nummod":
            if token.dep_ == "ammod" or token.dep_ == "advmmod":
                dep.append([str(token), token.dep_, list_paragraph.index(str(token))])
        mods.append(dep)
    return roots, mods


def main():
    global tokenizer_t5
    global result_list
    result_list = []
    model_checkpoint = "t5-base"
    batch_size = 2
    cache_dir = "/home/transformers_files/"
    cache_dir_models = cache_dir + "default_models/"
    cache_dir_tokenizers = cache_dir + "tokenizers/"

    train_set = LANG_Dataset('/home/student/nlpProj/data/train.labeled', flag_train=True)
    test_set = LANG_Dataset('/home/student/nlpProj/data/val.labeled',
                            flag_train=True)
    no_trans_set = LANG_Dataset('/home/student/nlpProj/data/val.unlabeled',
                                flag_train=True)
    comp_set = LANG_Dataset('/home/student/nlpProj/data/comp.unlabeled',
                            flag_train=True)

    en_nlp = spacy.load('en_core_web_sm')
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base")
    # load the text of the files
    train_set_eval = project_evaluate.read_file('data/train.labeled')
    test_set_eval = project_evaluate.read_file('data/val.labeled')
    # make hugging face datasets
    train_data = hug_face_dataset(train_set_eval).map(process_data, batched=True)
    train_data = train_data.map(lambda example: {"translation": {"eng": example["eng"], "ger": example["ger"]}},
                                remove_columns=["eng", 'ger'])
    test_data = hug_face_dataset(test_set_eval).map(process_data, batched=True)
    test_data = test_data.map(lambda example: {"translation": {"eng": example["eng"], "ger": example["ger"]}},
                              remove_columns=["eng", 'ger'])
    # defining the model
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, output_attentions=True)
    model_name = model_checkpoint.split("/")[-1]
    # define the trainer and train it

    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-German-to-English-Finalll", evaluation_strategy='epoch',
        learning_rate=0.0005,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        generation_max_length=210,
        predict_with_generate=True,
        gradient_accumulation_steps=16,
        push_to_hub=False,
        adafactor=True
        , fp16=True
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer_t5, model=model_t5)
    trainer = Seq2SeqTrainer(model_t5, args, train_dataset=train_data, eval_dataset=test_data,
                             data_collator=data_collator, tokenizer=tokenizer_t5, compute_metrics=compute_metrics_new)
    trainer.train()


if __name__ == '__main__':
    main()
