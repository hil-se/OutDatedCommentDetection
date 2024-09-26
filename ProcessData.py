import pandas as pd
import fasttext
from tqdm import tqdm

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
        sources_old = data["Source_Old"].tolist()
        targets_old = data["Target_Old"].tolist()
        sources_new = data["Source_New"].tolist()
        targets_new = data["Target_New"].tolist()
        for s in sources_old:
            f.write(filter(str(s)) + "\n")
        for t in targets_old:
            f.write(filter(str(t)) + "\n")
        for s in sources_new:
            f.write(filter(str(s)) + "\n")
        for t in targets_new:
            f.write(filter(str(t)) + "\n")
    print("\n")
    f.close()
    model = fasttext.train_unsupervised(w_filename, embedding, dim=dimensions)
    model_path = "Data/Trained_models/"+lang
    model.save_model(model_path + ".bin")


def generateEmbeddings(lang, embedding):
    model_path = "Data/Trained_models/" + lang
    ft = fasttext.load_model(model_path + ".bin")
    types_of_data = ["train", "valid", "test"]
    filename = "Data/Texts/" + lang

    for tod in types_of_data:
        print(lang, tod, filename)
        result = []
        data = pd.read_csv(filename + "_" + tod + ".csv")

        # Initialize tqdm progress bar
        for index, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {tod}"):
            s_old = str(row["Source_Old"])
            t_old = str(row["Target_Old"])
            s_new = str(row["Source_New"])
            t_new = str(row["Target_New"])

            # Skip rows with non-ASCII data
            if not (s_old.isascii() and t_old.isascii() and s_new.isascii() and t_new.isascii()):
                continue

            # Process and filter the text
            s_old = filter(s_old)
            t_old = filter(t_old)
            s_new = filter(s_new)
            t_new = filter(t_new)

            # Get sentence vectors from FastText
            s_old_vec = ft.get_sentence_vector(s_old)
            t_old_vec = ft.get_sentence_vector(t_old)
            s_new_vec = ft.get_sentence_vector(s_new)
            t_new_vec = ft.get_sentence_vector(t_new)

            # Append the result to the list
            result.append({
                "Source_Old": s_old_vec,
                "Target_Old": t_old_vec,
                "Source_New": s_new_vec,
                "Target_New": t_new_vec
            })

        # Write all the results to CSV at once after processing
        df = pd.DataFrame(result)
        df_path = "Data/" + embedding + "/embeddings/" + lang + "_" + tod
        df.to_csv(df_path + ".csv", index=False)


def processData(lang="java", embedding="cbow", dimensions=300):
    trainLanguageModel(lang, embedding, dimensions=dimensions)
    generateEmbeddings(lang, embedding)


processData()
