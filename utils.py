from __future__ import division
from __future__ import print_function
import random
from collections import namedtuple, Counter
import numpy as np

from BFS.KB import KB
from BFS.BFS import BFS
import os
import json
import threading

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = './datasets/NELL-995/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))

def compare(v1, v2):
    return sum(v1 == v2)

def teacher(e1, e2, num_paths, path):
	f = open(path) # dataPath + 'tasks/' + relation + '/' + 'graph.txt'
	content = f.readlines()
	f.close()
	kb = KB()
	for line in content:
		ent1, ent2, rel = line.rsplit()
		kb.addRelation(ent1, rel, ent2) # {entity1:[relation entity2]}
	# kb.removePath(e1, e2)
	intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths) #随机选择graph.txt中num_paths个不属于e1, e2的实体
	res_entity_lists = []
	res_path_lists = []
	for i in range(num_paths):
		#找e1, intermediates[i], e2之间的路径
		suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
		suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
		if suc1 and suc2:
			res_entity_lists.append(entity_list1 + entity_list2[1:])
			res_path_lists.append(path_list1 + path_list2)
	print('BFS found paths:', len(res_path_lists))

	# ---------- clean the path --------
	# res_entity_lists_new = []
	# res_path_lists_new = []
	paths = []
	for entities, relations in zip(res_entity_lists, res_path_lists):
		rel_ents = []
		for i in range(len(entities)+len(relations)):
			if i%2 == 0:
				rel_ents.append(entities[int(i/2)])
			else:
				rel_ents.append(relations[int(i/2)])

		#print(rel_ents)

		entity_stats = Counter(entities).items()
		duplicate_ents = [item for item in entity_stats if item[1]!=1]
		duplicate_ents.sort(key = lambda x:x[1], reverse=True)
		for item in duplicate_ents:
			ent = item[0]
			ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
			if len(ent_idx)!=0:
				min_idx = min(ent_idx)
				max_idx = max(ent_idx)
				if min_idx!=max_idx:
					rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
		entities_new = []
		relations_new = []
		for idx, item in enumerate(rel_ents):
			if idx%2 == 0:
				entities_new.append(item)
			else:
				relations_new.append(item)
		# print(entities_new)
		# print(relations_new)
		# print('....................')
		# res_entity_lists_new.append(entities_new)
		# res_path_lists_new.append(relations_new)
		paths.append([entities_new, relations_new])
	return paths

	# good_episodes = []
	# targetID = env.entity2id_[e2]
	# for path in zip(res_entity_lists_new, res_path_lists_new):
	# 	good_episode = []
	# 	for i in range(len(path[0]) -1):
	# 		currID = env.entity2id_[path[0][i]]
	# 		nextID = env.entity2id_[path[0][i+1]]
	# 		state_curr = [currID, targetID, 0]
	# 		state_next = [nextID, targetID, 0]
	# 		actionID = env.relation2id_[path[1][i]]
	# 		# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
	# 		# env.idx_state(state_curr): np.expand_dims(np.concatenate((curr, targ - curr)),axis=0)
	# 		good_episode.append(Transition(state = env.idx_state(state_curr), action = actionID, next_state = env.idx_state(state_next), reward = 1))
	# 	good_episodes.append(good_episode)
	# return good_episodes

def path_clean(path):
	rel_ents = path.split(' -> ')
	relations = []
	entities = []
	for idx, item in enumerate(rel_ents):
		if idx%2 == 0:
			relations.append(item)
		else:
			entities.append(item)
	entity_stats = Counter(entities).items()
	duplicate_ents = [item for item in entity_stats if item[1]!=1]
	duplicate_ents.sort(key = lambda x:x[1], reverse=True)
	for item in duplicate_ents:
		ent = item[0]
		ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
		if len(ent_idx)!=0:
			min_idx = min(ent_idx)
			max_idx = max(ent_idx)
			if min_idx!=max_idx:
				rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
	return ' -> '.join(rel_ents)

def prob_norm(probs):
	return probs/sum(probs)

def sample_path_each_task(relation, num_paths, graphpath):
	relationPath = dataPath + 'tasks/' + relation + '/' + 'train.pairs'
	f = open(relationPath)
	train_data = f.readlines()
	f.close()
	# print(train_data)
	sample_paths_dct = {}
	for i in range(len(train_data)):  # for each entity pair
		entity_pair = train_data[i].split(':')[0].split(',')
		if '-' in train_data[i].split(':')[1]:
			label = 0
		elif '+' in train_data[i].split(':')[1]:
			label = 1
		# label = train_data[i].split(':')[1]
		# print(entity_pair, label)
		e1 = entity_pair[0].split('$')[1]
		e2 = entity_pair[1].split('$')[1]
		sample_paths = teacher(e1, e2, num_paths, graphpath)
		# print((e1, e2))
		# print(sample_paths)
		# print(label)
		# print('.......................')
		sample_paths_dct[i] = {'origin entity pair': [e1, e2], 'sample_paths': sample_paths, 'label': label}
		# print(sample_paths_dct)
		print(relation, i)
		print('.......................')
	jsonPath = dataPath + 'tasks/' + relation + '/' + 'sample_paths.json'
	with open(jsonPath, 'w') as obj:
		json.dump(sample_paths_dct, obj, indent=4)

def sample_path():
	# graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
	# relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
	tasks = os.listdir(dataPath + 'tasks/')
	graphpath = dataPath + 'kb_env_rl.txt'
	# print(tasks)
	# relation = 'concept_agentbelongstoorganization'
	threads = []
	for relation in tasks:
		threads.append(threading.Thread(target=sample_path_each_task, args=(relation, 5, graphpath)))
		# sample_path_each_task(relation=relation, num_paths=5, graphpath=graphpath)
	for t in threads:
		t.start()



if __name__ == '__main__':
	sample_path()
	# print(prob_norm(np.array([1,1,1])))
	#path_clean('/common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01d34b -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/0lfyx -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01y67v -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/028qyn -> /people/person/nationality -> /m/09c7w0')





