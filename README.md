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
All the newly created csv files will have two columns: 'Source' carrying the comment, and 'Target' carrying the code.
### Dependencies
The required dependencies must be installed to run the source code.
```
pip install -r requirements.txt
```
### Generating Embeddings
run the ProcessData.py file.
```
python ProcessData.py
```
Running this file will produce new java_train.csv, java_test.csv, and java_valid.csv files. In the Source and Target columns of these files, instead of storing actual comment and code, it will store embeddings of comment and code. These files will be stored in the newly created directory Data/embedding/CodeSearch300.      

### The Process of Training
![SWEN 732 Documentation - domain model (1)](https://github.com/user-attachments/assets/edea50b3-00b2-45e1-8d7a-594b13d4aed1)


