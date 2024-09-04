# OutDatedCommentDetection

### Dataset
You can download the dataset used in this project from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi). This dataset was created as part of the CUP2 project whose replication kit can be found [here](https://github.com/Tbabm/CUP2). Its a large-scale dataset with over 4 million Java code-comment change samples. It mainly consists of method-level code changes and updates on method header comments. 
**Instructions**:
- Download cup2_dataset.zip and extract in Data/Texts directory  
- Move the contents of cup2_dataset directory to Data/Texts. 
- Run train_jasonl_to_csv.py to create java_train.csv file.  
- Run test_jasonl_to_csv.py to create java_test.csv file.
- Run valid_jasonl_to_csv.py to create java_valid.csv file.
```bash
mv cup2_dataset.zip Data/Texts
cd Data
cd Texts
unzip cup2_dataset.zip
cd cup2_dataset
mv Java_train.jsonl Java_test.jsonl Java_valid.jsonl ../
```
