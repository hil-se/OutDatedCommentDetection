# OutDatedCommentDetection

### Dataset
You can download the dataset used in this project from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi). This dataset was created as part of the CUP2 project whose replication kit can be found [here](https://github.com/Tbabm/CUP2). Its a large-scale dataset with over 4 million Java code-comment change samples. It mainly consists of method-level code changes and updates on method header comments. 
**Instructions**:
- Download cup2_dataset.zip and extract in Data/Texts directory  
- Move the contents of cup2_dataset directory to Data/Texts. 
- Run train_jasonl_to_csv.py to create java_train_all.csv file.  
- Run test_jasonl_to_csv.py to create java_test_all.csv file.
- Run valid_jasonl_to_csv.py to create java_valid_all.csv file.
- Run train_filter_true_labels.py to create java_train.csv file.
- Run test_filter_true_labels.py to create java_test.csv file.
- Run valid_filter_true_labels.py to create java_valid.csv file.
```bash
mv cup2_dataset.zip Data/Texts
cd Data
cd Texts
unzip cup2_dataset.zip
cd cup2_dataset
mv Java_train.jsonl Java_test.jsonl Java_valid.jsonl ../
```
All the newly created csv files will have five columns: Source_old (carrying the comment of previous commit), Target_old (carrying the method of previous commit), Source_new (carrying the comment of new commit), Target_new (carrying the method of new commit), and Label. The Label is true if Source_old is outdated (differs significantly from Source_new) and false if it is up-to-date (similar to Source_new).

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
![SWEN 732 Documentation](https://github.com/user-attachments/assets/cb3392d4-b455-4892-97ba-41dc01474acd)

- Encoder 1 only takes method embeddings as input
- Encoder 2 only takes comment embeddings as input
- First we train the two encoders to consider the cosine similarity between old_comment and old_method as 1
- Then we train the two encoders to consider the cosine similarity between old_comment and new_method where the label was false as 1
- Case 1: We train the two encoders to consider the cosine similarity between old_comment and new_method where the label was true as 0
- Case 2: We train the two encoders to consider the cosine similarity between old_comment and new_method same as the cosine similarity between the old_comment and new_comment. Because we assumed that the new comment should reflect the changes that the new method carries.


