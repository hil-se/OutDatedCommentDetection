import pandas as pd
import fasttext

to_remove = [",", ".", "<", ">", "?", "/", ";", ":", "'", "!", "#", "$", "%", "^", "~",
             "*", "(", ")", "{", "}", "[", "]", "\\", "-", "_", "\n", "\t" "@", "&", "`"]

def camel_case_split(text):
    text_list = text.splitlines()
    res = ""
    for s in text_list:
        idx = list(map(str.isupper, s))
        l = [0]
        for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
            if x and not y:
                l.append(i)
            elif not x and y:
                l.append(i+1)
        l.append(len(s))
        words = [s[x:y] for x, y in zip(l, l[1:]) if x < y]
        sentence = ' '.join(words)
        while "  " in sentence:
          sentence = sentence.replace("  ", " ")
        res = res + sentence +"\n"
    res = res.rstrip()
    return res

def filter(text):
  text = camel_case_split(text)
  for symbol in to_remove:
    text = text.replace(symbol, " ")
  while "  " in text:
    text = text.replace("  ", " ")
  return text


def trainLanguageModel(lang, embedding, dimensions=300):
    w_filename = "Data/Texts/"+lang+".txt"
    f = open(w_filename, "w+")
    types_of_data = ["train"]
    print(lang)
    for tod in types_of_data:
        print(tod)
        filename = lang + "_" + tod + ".csv"
        data = pd.read_csv("Data/Texts/" + filename)
        sources = data["Source"].tolist()
        targets = data["Target"].tolist()
        for s in sources:
            f.write(filter(str(s)) + "\n")
        for t in targets:
            f.write(filter(str(t)) + "\n")
    print("\n")
    f.close()
    model = fasttext.train_unsupervised(w_filename, embedding, dim=dimensions)
    model_path = "Data/Trained models/"+lang
    model.save_model(model_path + ".bin")

def generateEmbeddings(lang, embedding):
    model_path = "Data/Trained models/" + lang
    ft = fasttext.load_model(model_path+".bin")
    types_of_data = ["train", "valid", "test"]
    result = []
    filename = "Data/Texts/" + lang
    for tod in types_of_data:
        print(lang, tod, filename)
        result = []
            data = pd.read_csv(filename + "_" + tod + ".csv")
            for index, row in data.iterrows():
                s = str(row["Source"])
                t = str(row["Target"])
                if s.isascii() == False or t.isascii() == False:
                    continue
                s = filter(s)
                t = filter(t)
                s = ft.get_sentence_vector(s)
                t = ft.get_sentence_vector(t)
                result.append({"Source": t, "Target": s})
                df = pd.DataFrame(result)
                df_path = "Data/" + embedding + "/CodeSearch300/" + lang + "_" + tod
                df.to_csv(df_path + ".csv", index=False)


def processData(lang="java", embedding="cbow", dimensions=300):
    trainLanguageModel(lang, embedding, dimensions=dimensions)
    generateEmbeddings(lang, embedding)


processData()
