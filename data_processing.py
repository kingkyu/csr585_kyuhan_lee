import sqlite3
conn = sqlite3.connect('fever.db')
cursor = conn.cursor()

# create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS evidence_index(
	id text PRIMARY KEY,
	file_num integer NOT NULL,
	line_num integer NOT NULL
	)
	''')

for i in range(109):
	print(i)
	file_num = i+1
	if file_num < 10:
		file = open("wiki-00"+str(i+1)+".jsonl", "r")
	elif file_num < 100:
		file = open("wiki-0"+str(i+1)+".jsonl", "r")
	else:
		file = open("wiki-"+str(i+1)+".jsonl", "r")
	line_num = 0
	for line in file:
		temp_list = line.split(":")
		_id = temp_list[1].replace("\", \"text\"","").replace("\"","").replace(" ","")
		line_num += 1
		cursor.execute("INSERT OR IGNORE INTO evidence_index(ID, file_num, line_num) VALUES(?, ?, ?)", (_id, file_num, line_num))
		conn.commit()