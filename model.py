import json
import random
import sqlite3
import math
from scipy.sparse import hstack
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# connecting to sqlite3 db where indices for wikipedia pages are stored
# the indices have been stored using "data_processing.py" included in the repository
# these indices will be used when retrieving relevant data for fact checking tasks
conn = sqlite3.connect('fever.db')
cursor = conn.cursor()

# a process for collecing index information of gold standard evidence at a documnet level 
# it is for extracting a vocabulary, from the training data, for the word vector (i.e., 5000 most frequently appearing words) used in this algorithm
train_file = open('train.jsonl', "r") # open a file containing training data
train_wiki_page_index_set = set() # a set-typed variable to store the indices of wiki pages in the training data
for line in train_file:
	json_data = json.loads(line) # parse each line (or data point) as a json-typed element
	if json_data["label"] == "NOT ENOUGH INFO": # data points labeled as "NOT ENOUGH INFO" do not have corresponding gold standard documents, and thus no use for extracting the word vector
		continue
	for i in json_data["evidence"]:
		for j in i:
			cursor.execute("SELECT file_num, line_num FROM evidence_index WHERE id = ?", (j[2],)) # from the index db, we select the file number and the line number for correspodning document
			# exception handling for spanish characters with accents
			# this issue will be resolved in a later version
			try:
				temp = cursor.fetchone()
				file_num = temp[0]
				line_num = temp[1]
				train_wiki_page_index_set.add((file_num,line_num))
			except:
				continue
print("done extracting an initial set of vocabulary")

training_data_for_vocabulary = []
for element in train_wiki_page_index_set:
	file_num = element[0]
	line_num = element[1]
	if file_num < 10:
		file = open("wiki-00"+str(element[0])+".jsonl", "r")
	elif file_num < 100:
		file = open("wiki-0"+str(element[0])+".jsonl", "r")
	else:
		file = open("wiki-"+str(element[0])+".jsonl", "r")
	line_num_count = 1
	for line in file:
		if line_num_count == line_num:
			json_data = json.loads(line)
			training_data_for_vocabulary.append(json_data["text"])
			break # break the loop for searching documnets when the targeted one is retrieved
		else:
			pass
		line_num_count += 1

# Since, in the original baseline paper, the authors used both TF and TF-IDF vectors, we estimate them separtely
tfvectorizer = TfidfVectorizer(stop_words='english', max_features=5000, use_idf = False) # vectorizer for TF
tfidfvectorizer = TfidfVectorizer(stop_words='english', max_features=5000, use_idf = True) # vectorizer for TF-IDF
tfvectorizer.fit(training_data_for_vocabulary)
tfidfvectorizer.fit(training_data_for_vocabulary)
print("done word vectorization")

# prepare training data

# count the number of support-labeled, refute-labeled, and not-enough-info-labled claims for a post-anlaysis
support_count = 0
refute_count = 0
not_enough_info_count = 0

# each input contains the TF vectors of a claim and its evidences in a concatenated manner
# and the cosine similarity of the TF-IDF vectors of the claim and the evidences
X_train = []
# each output can be classified into 0=support, 1=refute, and 2=not-enough-info
y_train = []
exception_count = 0
train_file = open('train.jsonl', "r")
for line in train_file:
	json_data = json.loads(line)
	claim_label = json_data["label"]
	# vecotrize a claim with the pretrained TF and TF-IDF vectors
	claim_tf = tfvectorizer.transform([json_data["claim"]]) 
	claim_tfidf = tfidfvectorizer.transform([json_data["claim"]])
	try:
		evidence_text = []
		# for each claim whose gold label is "NOT ENOUGH INFO", we randonly generate three evidences since they do not have gold evidences
		if json_data["label"] == "NOT ENOUGH INFO":
			for i in range(3):
				random_file_num = random.randint(1,109)
				if random_file_num < 10:
					evidence_file = open("wiki-00"+str(random_file_num)+".jsonl", "r")
				elif random_file_num < 100:
					evidence_file = open("wiki-0"+str(random_file_num)+".jsonl", "r")
				else:
					evidence_file = open("wiki-"+str(random_file_num)+".jsonl", "r")
				max_count = 0
				for i in evidence_file:
					max_count += 1
				random_line_num = random.randint(0, max_count)

				if random_file_num < 10:
					evidence_file = open("wiki-00"+str(random_file_num)+".jsonl", "r")
				elif random_file_num < 100:
					evidence_file = open("wiki-0"+str(random_file_num)+".jsonl", "r")
				else:
					evidence_file = open("wiki-"+str(random_file_num)+".jsonl", "r")
				count = 1
				for evidence_line in evidence_file:
					if count == random_line_num:
						evidence_data = json.loads(evidence_line)
						evidence_text.append(evidence_data["text"])
					else:
						pass
					count += 1
		# for each claim whose gold label is either "SUPPORT" or "REFUTE", we retreived gold documents
		else:
			for i in json_data["evidence"]:
				for j in i:
					cursor.execute("SELECT file_num, line_num FROM evidence_index WHERE id = ?", (j[2],))	
					temp = cursor.fetchone()
					file_num = temp[0]
					line_num = temp[1]
					if file_num < 10:
						evidence_file = open("wiki-00"+str(file_num)+".jsonl", "r")
					elif file_num < 100:
						evidence_file = open("wiki-0"+str(file_num)+".jsonl", "r")
					else:
						evidence_file = open("wiki-"+str(file_num)+".jsonl", "r")
					count = 1
					for evidence_line in evidence_file:
						if count == line_num:
							evidence_data = json.loads(evidence_line)
							evidence_text.append(evidence_data["text"])
						else:
							pass
						count += 1
		# concantenate multiple evidences
		evidence_concat = ""
		for ev in evidence_text:
			evidence_concat += (" " + ev)
		# vecotrize evidences with the pretrained TF and TF-IDF vectors
		evidence_tf = tfvectorizer.transform([evidence_concat])
		evidence_tfidf = tfidfvectorizer.transform([evidence_concat])
		# estimate TF-IDF cosine similarity of the claim and the evidence vetors
		tfidf_cosine_similarity = 1 - spatial.distance.cosine(claim_tfidf.toarray(), evidence_tfidf.toarray())
		if math.isnan(tfidf_cosine_similarity):
			tfidf_cosine_similarity = 0
		# concatenation input vectors into one
		tf_vector = hstack((claim_tf, evidence_tf))
		input_vector = hstack((tf_vector, [tfidf_cosine_similarity]))
		# append the final input vector
		X_train.append(input_vector.toarray()[0]) # take the first element since the "toarray()" returns 2-dim data strucute when 1-dim is needed
		# append the corresponding output
		if claim_label == "SUPPORTS":
			y_train.append(0)
			support_count += 1
		elif claim_label == "REFUTES":
			y_train.append(1)
			refute_count += 1
		else:
			y_train.append(2)
			not_enough_info_count += 1
	except Exception as e:
		print(e)
		print(json_data["claim"])
		exception_count += 1

print("done training data preparation")

# train a multi-layer perceptron
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

print("done mlp model fitting")


# prepare test data
# same process for preparing training data
# for understanding the codes, refer to the comments on the codes for training data preparation
test_support_count = 0
test_refute_count = 0
test_not_enough_info_count = 0
X_test = []
y_test = []
test_exception_count = 0
test_file = open('test.jsonl', "r")
for line in test_file:
	json_data = json.loads(line)
	claim_label = json_data["label"]
	claim_tf = tfvectorizer.transform([json_data["claim"]])
	claim_tfidf = tfidfvectorizer.transform([json_data["claim"]])
	try:
		evidence_text = []
		if json_data["label"] == "NOT ENOUGH INFO":
			for i in range(3):
				random_file_num = random.randint(1,109)
				if random_file_num < 10:
					evidence_file = open("wiki-00"+str(random_file_num)+".jsonl", "r")
				elif random_file_num < 100:
					evidence_file = open("wiki-0"+str(random_file_num)+".jsonl", "r")
				else:
					evidence_file = open("wiki-"+str(random_file_num)+".jsonl", "r")
				max_count = 0
				for i in evidence_file:
					max_count += 1
				random_line_num = random.randint(0, max_count)

				if random_file_num < 10:
					evidence_file = open("wiki-00"+str(random_file_num)+".jsonl", "r")
				elif random_file_num < 100:
					evidence_file = open("wiki-0"+str(random_file_num)+".jsonl", "r")
				else:
					evidence_file = open("wiki-"+str(random_file_num)+".jsonl", "r")
				count = 1
				for evidence_line in evidence_file:
					if count == random_line_num:
						evidence_data = json.loads(evidence_line)
						evidence_text.append(evidence_data["text"])
					else:
						pass
					count += 1
		else:	
			for i in json_data["evidence"]:
				for j in i:
					cursor.execute("SELECT file_num, line_num FROM evidence_index WHERE id = ?", (j[2],))	
					temp = cursor.fetchone()
					file_num = temp[0]
					line_num = temp[1]
					if file_num < 10:
						evidence_file = open("wiki-00"+str(file_num)+".jsonl", "r")
					elif file_num < 100:
						evidence_file = open("wiki-0"+str(file_num)+".jsonl", "r")
					else:
						evidence_file = open("wiki-"+str(file_num)+".jsonl", "r")
					count = 1
					for evidence_line in evidence_file:
						if count == line_num:
							evidence_data = json.loads(evidence_line)
							evidence_text.append(evidence_data["text"])
						else:
							pass
						count += 1
		evidence_concat = ""
		for ev in evidence_text:
			evidence_concat += (" " + ev)
		evidence_tf = tfvectorizer.transform([evidence_concat])
		evidence_tfidf = tfidfvectorizer.transform([evidence_concat])
		tfidf_cosine_similarity = 1 - spatial.distance.cosine(claim_tfidf.toarray(), evidence_tfidf.toarray())
		if math.isnan(tfidf_cosine_similarity):
			tfidf_cosine_similarity = 0		
		tf_vector = hstack((claim_tf, evidence_tf))
		input_vector = hstack((tf_vector, [tfidf_cosine_similarity]))
		X_test.append(input_vector.toarray()[0])

		if claim_label == "SUPPORTS":
			y_test.append(0)
			test_support_count += 1
		elif claim_label == "REFUTES":
			y_test.append(1)
			test_refute_count += 1
		else:
			y_test.append(2)
			test_not_enough_info_count += 1
	except Exception as e:
		print(e)
		print(json_data["claim"])
		test_exception_count += 1
print("done test data preparation")

#print(exception_count, test_exception_count)
print("Results")
print("Distribution of labels in the training data (support, refute, indeterminable) ", support_count, refute_count, not_enough_info_count) # distribution of outcomes in training data
print("Distribution of labels in the testdata (support, refute, indeterminable) ", test_support_count, test_refute_count, test_not_enough_info_count) # distribution of outcomes in test datda
print("Overall Accuracy:", mlp.score(X_test, y_test)) # accuary of the model 
# for error analysis, write out prediction restuls in a separate text file
X_predicted = mlp.predict(X_test)
result = open("result.txt", "w")
for i in range(len(X_predicted)):
	result.write(str(X_predicted[i]) + "\t" + str(y_test[i]) + "\n")
