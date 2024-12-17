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

- run the Training.py file

![SWEN 732 Documentation](https://github.com/user-attachments/assets/cb3392d4-b455-4892-97ba-41dc01474acd)

- Encoder 1 only takes method embeddings as input
- Encoder 2 only takes comment embeddings as input
- First we train the two encoders to consider the cosine similarity between old_comment and old_method as 1
- Then we train the two encoders to consider the cosine similarity between old_comment and new_method where the label was false as 1
- Case 1: We train the two encoders to consider the cosine similarity between old_comment and new_method where the label was true as 0
- Case 2: We train the two encoders to consider the cosine similarity between old_comment and new_method same as the cosine similarity between the old_comment and new_comment. Because we assumed that the new comment should reflect the changes that the new method carries.

### Testing
- run the Testing.py file
- This file creates a new CSV file 'predicted_cosine_similarity_results.csv', which has three columns predicted_similarity (carries the cosine similarity between Source_old and Target_new predicted by the model), true_labels (value from the Label column), predicted_labels (our predicted label). 
- We used rank_error method that computes the average rank error for instances with Label = True.
- It sorts the data by Cosine Similarity in ascending order which in the ideal scenario will arrange all the true labels before all the false labels.
- For each True Label, it accumulates the count of False labels encountered before it, normalized by the total number of entries.
- It divides the total accumulated error by the count of True labels, giving the average rank error.
- We tested our model on java_valid_all.csv (an unbalanced real world set) and then we tested it on a balanced set as well called java_valid_all_subset.csv. java_valid_all_subset carries 500 True label instances and 500 False label instances.
  
![image](https://github.com/user-attachments/assets/52245b09-8f3f-4ba9-9b9c-0be8058408d8)

Results: Rank-error=0.3564, F1-Score=0.026, Recall=0.019, Precision=0.042
![395314628-b1bd49d2-7665-4805-9002-e8e206a6aad5](https://github.com/user-attachments/assets/e9cbf8c8-5568-4ce0-a304-c7eea248e3b3)





