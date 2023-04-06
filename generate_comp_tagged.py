from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    Adafactor
import project_evaluate
from transformers import T5Tokenizer


def tagger(unlabeled_file, pred):
    """
    The function gets an unlabeled file and adds labels(translation) according to the required format
    :param unlabeled_file: the unlabeled file to be labeled
    :param pred: the tagged gotten out of the model

    """
    with open('data/' + str(unlabeled_file) + '.unlabeled', encoding="utf8") as f:
        contents = f.read()

    # write predictions in predictions file
    pred_used = 0
    with open(str(unlabeled_file) + '_322995358_318170917.labeled', 'w', encoding="utf8") as file:
        for i in range(len(contents.split('\n'))):
            line = contents.split('\n')[i]
            if line.strip():
                # writes the translation as was predicted by the model
                if 'Roots in English:' in line:
                    file.write('English:')
                    file.write('\n')
                elif 'Modifiers in English:' in line:
                    file.write(pred[pred_used])
                    file.write('\n')
                    pred_used += 1
                # writes the german part as it was before
                if 'Roots in English:' not in line and 'Modifiers in English:' not in line:
                    file.write(line)
                    file.write('\n')

            else:
                file.write('\n')

def comp_tagger(pred):
    """
    The function get the prediction for the translation and labels the comp file
    :param pred: the predicted translation of the model
    """
    tagger('comp',pred)


def val_tagger(pred):
    """
     The function get the prediction for the translation and labels the val file
    :param pred: the predicted translation of the model

    """
    tagger('val', pred)

def main():
    global tokenizer_t5

    model_checkpoint = r"checkpoint-3000/"
    cache_dir = "/home/transformers_files/"
    cache_dir_models = cache_dir + "default_models/"
    cache_dir_tokenizers = cache_dir + "tokenizers/"

    # load the tokenizer
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base")

    # read the files to be labeled
    unlabeled_set_eval = project_evaluate.read_file('data/val.unlabeled')
    comp_unlabeled = project_evaluate.read_file('data/comp.unlabeled')

    # split the comp file into part
    split_size = len(comp_unlabeled[1]) // 10
    splits = [comp_unlabeled[1][:split_size], comp_unlabeled[1][split_size:2 * split_size],
              comp_unlabeled[1][2 * split_size:3 * split_size], comp_unlabeled[1][3 * split_size:4 * split_size],
              comp_unlabeled[1][4 * split_size:5 * split_size], comp_unlabeled[1][5 * split_size:6 * split_size],
              comp_unlabeled[1][6 * split_size:7 * split_size], comp_unlabeled[1][7 * split_size:8 * split_size],
              comp_unlabeled[1][8 * split_size:9 * split_size], comp_unlabeled[1][9 * split_size:]]
    # tokenizing the different parts
    splits_tokinized = []
    for cur_split in splits:
        splits_tokinized.append(
            tokenizer_t5(["translate German to English: " + sen.split('Roots in English')[0] for sen in cur_split],
                         max_length=220, truncation=True, padding=True,
                         return_tensors="pt").input_ids)

    # load the model for the checkpoint
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, output_attentions=True)

    # tokenizing the val file and predict
    toknized_val = tokenizer_t5(
        ["translate German to English: " + sen.split('Roots in English')[0] for sen in unlabeled_set_eval[1]],
        max_length=220, truncation=True, padding=True, return_tensors="pt").input_ids
    output = model_t5.generate(toknized_val, max_length=300)

    # decode the predictions and tagging val file
    predictions_val = tokenizer_t5.batch_decode(output, skip_special_tokens=True)
    val_tagger(predictions_val)


    # tagging comp file
    predictions_comp = []
    i = 0
    for cur_split in splits_tokinized:
        output_comp = model_t5.generate(cur_split, max_length=300)
        print(i)
        decoded_comp = tokenizer_t5.batch_decode(output_comp, skip_special_tokens=True)

        predictions_comp.extend(decoded_comp)
        i += 1
    comp_tagger(predictions_comp)



main()
