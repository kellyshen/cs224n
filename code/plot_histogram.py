import numpy as np
import matplotlib.pyplot as plt

questions_f = open('../data/train.question', 'r')
contexts_f = open('../data/train.context', 'r')
answers_f = open('../data/train.answer', 'r')
spans_f = open('../data/train.span', 'r')

questions = []
contexts = []
answers = []
starts = []
ends = []

len_questions_lookup = {} 
len_contexts_lookup = {}
len_answers_lookup = {} # counts of length of answer (raw)
len_answers_lookup_norm = {} # counts of length of answer (relative to context len)
len_answers_lookup_norm_q = {} # counts of length of answer (relative to question len)
starts_lookup_raw = {} # counts of where answer starts (raw index)
ends_lookup_raw = {} # counts of where answer ends (raw index)
starts_lookup_norm = {} # counts of where answer starts (relative to context len)
ends_lookup_norm = {} # counts of where answer ends (relative to context len)

len_questions = []
len_contexts = []
len_answers = []
starts_raw = []
ends_raw = []
len_answers_norm = []
len_answers_norm_q = []
starts_norm = []
ends_norm = []

for line in questions_f:
	curr_question = line.split(" ")
	questions.append(curr_question)
	len_questions.append(len(curr_question))
	#curr_count = len_questions_lookup.get(len(curr_question), 0)
	#curr_count+=1
for line in contexts_f:
	curr_context = line.split(" ")
	contexts.append(curr_context)
	len_contexts.append(len(curr_context))
	#curr_count = len_contexts_lookup.get(len(curr_context), 0)
	#curr_count+=1
for line in answers_f:
	curr_answer = line.split(" ")
	answers.append(curr_answer)
	len_answers.append(len(curr_answer))
	#curr_count = len_answers_lookup.get(len(curr_answer), 0)
	#curr_count+=1
for line in spans_f:
	start, end = line.split(" ")
	starts.append(int(start))
	ends.append(int(end))
	starts_raw.append(int(start))
	ends_raw.append(int(end))
	#curr_count_start = starts_lookup_raw.get(int(start), 0)
	#curr_count_start+=1
	#curr_count_end = ends_lookup_raw.get(int(end), 0)
	#curr_count_end+=1

for i, answer in enumerate(answers):
	# answer len relative to context len
	relative_len = len(answer) / float(len(contexts[i])) 
	#curr_count_context = len_answers_lookup_norm.get(relative_len, 0)
	#curr_count+=1
	len_answers_norm.append(relative_len)
	# answer start / end pos relative to context len 
	relative_start = starts[i] / float(len(contexts[i])) 
	#curr_count_start = starts_lookup_norm.get(relative_start, 0)
	#curr_count_start +=1
	starts_norm.append(relative_start)
	relative_end = ends[i] / float(len(contexts[i]))
	#curr_count_end = ends_lookup_norm.get(relative_end, 0)
	#curr_count_end+=1
	ends_norm.append(relative_end)
	# answer len relative to question len 
	#print 'len answer ', len(answer)
	#print 'len question ', len(questions[i])
	relative_len_q = len(answer) / float(len(questions[i]))
	#print 'relative len ', relative_len_q
	#curr_count_answer = len_answers_lookup_norm_q.get(relative_len_q, 0)
	#curr_count_answer+=1
	len_answers_norm_q.append(relative_len_q)
	# answer len relative to start pos 
	# answer len relative to normalized start pos 

titles = ['answer start indices', 'answer end indices', 'answer start indices normalized by context lengths', 'answer end indices normalized by context lengths']
#titles = ['question lengths', 'context lengths', 'answer lengths raw', 'answer lengths normalized by context lengths', 'answer lengths normalized by question lengths', 'answer start indicies', 'answer end indicies', 'answer start indicies normalized by context lengths', 'answer end indicies normalized by context lengths']
for i, data in enumerate([len_questions, len_contexts, len_answers, len_answers_norm, len_answers_norm_q, starts_raw, ends_raw, starts_norm, ends_norm]): #[len_questions_lookup, len_contexts_lookup, len_answers_lookup, len_answers_lookup_norm, len_answers_lookup_norm_q, starts_lookup_raw, ends_lookup_raw, starts_lookup_norm, ends_lookup_norm]:
	#hist, bins = np.histogram(data)#, density='True')
	#plt.plot(bins[:-1], hist)
	#plt.title(titles[i] + ' np')
	#plt.show()
	plt.hist(data, bins='auto', density='True')
	plt.title(titles[i])
	plt.savefig(titles[i])
	plt.show()
	



