import json
import os
import numpy as np
import random
import math
# import tensorflow as tf

class DataLoader():
    def __init__(self, dataset_name, task, batch_size, shuffle=False, max_path_mun=10, split_path_num=10):
        self.dataset_name = dataset_name
        self.task = task
        self.batch_size = batch_size
        self.max_path_mun = max_path_mun
        self.split_path_num = split_path_num
        self.base_path = './datasets/{}'.format(dataset_name)

        self.entity_embedding_table = np.loadtxt('{}/{}'.format(self.base_path, 'entity2vec.bern'))
        self.entity_ids, self.entity_num = self.entity2id()
        self.relation_embedding_table = np.loadtxt('{}/{}'.format(self.base_path, 'relation2vec.bern'))
        self.relation_ids, self.relation_num = self.relation2id()#######
        # print(self.relation_ids)

        self.embedding_table = np.concatenate([self.entity_embedding_table,
                                               self.relation_embedding_table], axis=0) #,np.zeros((1, self.relation_embedding_table.shape[-1]))
        # self.total_idx = int(self.embedding_table.shape[0])
        self.entity_type_dct = self.get_entityType_ids() ##########----------
        # print(self.entity_type_dct)
        self.max_train_path_len = 0
        # {'origin_entity_pair': [e_head_id, e_tail_id],
        #  'sample_paths': [[e1,r1,e2,r2,e3,...], [],...],
        #  'task_id': relation_id]}
        self.train_samples = self.create_train_sample_dct()
        self.train_samples_num = len(self.train_samples)
        self.batch_num_train = math.ceil(self.train_samples_num/self.batch_size)
        self.sample_ids_train = list(range(self.train_samples_num))
        if shuffle:
            random.shuffle(self.sample_ids_train)

        self.max_test_path_len = 0
        self.test_samples = self.create_test_sample_dct()
        self.test_samples_num = len(self.test_samples)
        self.batch_num_test = math.ceil(self.test_samples_num / self.batch_size)
        self.sample_ids_test = list(range(self.test_samples_num))

        # 在temp文件夹中，ent2id.pkl是实体id，entity_type.npy是每个实体id对应的实体类型id，type2id.pkl是每种实体类型id对应的类型名称；
        # 在temp文件夹中，rel2id.pkl是关系id，relation_type.npy是每个关系id对应的关系类型id
        # temp文件夹中的实体id和关系id与上面处理的结果不对应，需要额外处理
        # self.type2IDs = np.load('{}/temp/{}'.format(self.base_path, 'type2id.pkl'), allow_pickle=True)
        # self.entityType_embedding = np.load('{}/temp/{}'.format(self.base_path, 'entity_type.npy'), allow_pickle=True)
        # self.qqq = np.load('{}/temp/{}'.format(self.base_path, 'rel2id.pkl'), allow_pickle=True)
        # print(self.type2IDs)
        # print(self.entityType_embedding)
        # print((self.qqq))

        # self.relation_type_dct = self.get_relationType_ids()

    def get_entityType_ids(self):

        if self.dataset_name == 'NELL-995':
            self.entity_type_num = 267
            self.max_type_num_per_entity = 16

            temp_entity_ids = np.load('{}/temp/{}'.format(self.base_path, 'ent2id.pkl'), allow_pickle=True)
            temp_entity_type_dct = np.load('{}/temp/{}'.format(self.base_path, 'entity_type.npy'), allow_pickle=True)

            entity_type_dct = {}
            temp_entity_ids_keys = temp_entity_ids.keys()
            # c = 0
            # max_type_num_per_entity = 0
            # max_entity_type_id = 0
            # print(temp_entity_ids_keys)
            for i, k in enumerate(self.entity_ids):
                if k in temp_entity_ids_keys:
                    temp_entity_id = temp_entity_ids[k]
                    # print(i, self.entity_ids[k])
                    entity_type_dct[k] = temp_entity_type_dct[temp_entity_id]
                    # if max(temp_entity_type_dct[temp_entity_id]) > max_entity_type_id:
                    #     max_entity_type_id = max(temp_entity_type_dct[temp_entity_id])
                    # if len(temp_entity_type_dct[temp_entity_id]) > max_type_num_per_entity:
                    #     max_type_num_per_entity = len(temp_entity_type_dct[temp_entity_id])
                else:
                    # print(k)
                    # c += 1
                    entity_type_dct[k] = [self.entity_type_num]
                    # print(len(self.entity_ids))
            # print(len(temp_entity_ids))
            # print(len(entity_type_dct))
            # print(entity_type_dct)
            # print(c)
        elif self.dataset_name == 'FB15k-237':
            # self.entity_type_num = 4054
            # self.max_type_num_per_entity = 138

            entity_type_dct = {}  # {entyte_name:[type1,type2,type3,...]}
            entity_type_dct_temp = {}
            type2id = {}
            self.entity_type_num = 0 # 4054
            self.max_type_num_per_entity = 0 #138

            with open('{}/{}/{}'.format(self.base_path, 'type_information', 'entity2type.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    type_list = []
                    line = line.split('\n')[0].split('\t')
                    entity_name = line[0]
                    type_set = line[1:]
                    # if entity_name == '/m/09c7w0':
                    #     print(type_set, len(type_set))
                    for type in type_set:
                        if type not in type2id.keys():
                            type2id[type] = self.entity_type_num
                            self.entity_type_num += 1
                        type_list.append(type2id[type])
                        entity_type_dct_temp[entity_name] = type_list

                    if len(type_list) > self.max_type_num_per_entity:
                        self.max_type_num_per_entity = len(type_list)
                    # if len(type_list) == self.max_type_num_per_entity:
                    #     print(entity_name, type_list)
                    # print(entity_name, type_list, len(type_list), '+', len(type_set))
            # print(self.entity_type_num, self.max_type_num_per_entity)

            for i, k in enumerate(self.entity_ids):
                if k in entity_type_dct_temp.keys():
                    entity_type_dct[k] = entity_type_dct_temp[k]
                else:
                    entity_type_dct[k] = [self.entity_type_num]
            # print(entity_type_dct)

        return entity_type_dct

    # def get_relationType_ids(self):
    #
    #     if self.dataset_name == 'NELL-995':
    #         self.relation_type_num = 267
    #         self.max_type_num_per_relation = 255
    #
    #     temp_relation_ids = np.load('{}/temp/{}'.format(self.base_path, 'rel2id.pkl'), allow_pickle=True)
    #     temp_relation_type_dct = np.load('{}/temp/{}'.format(self.base_path, 'relation_type.npy'), allow_pickle=True)
    #
    #     print(temp_relation_ids)
    #     # print(len(self.relation_ids))
    #
    #     relation_type_dct = {}
    #     temp_relation_ids_keys = temp_relation_ids.keys() # 'reverse'
    #
    #     max_type_num_per_relation = 0
    #     max_relation_type_id = 0
    #     for i, k in enumerate(self.relation_ids):# 'inv'
    #         if 'inv' in k:
    #             k = k.replace('inv', 'reverse')
    #         if k in temp_relation_ids_keys:
    #             temp_relation_id = temp_relation_ids[k]
    #             # print(i, self.entity_ids[k])
    #             if 'reverse' in k:
    #                 k = k.replace('reverse', 'inv')
    #             relation_type_dct[k] = temp_relation_type_dct[temp_relation_id]
    #             if max(temp_relation_type_dct[temp_relation_id]) > max_relation_type_id:
    #                 max_relation_type_id = max(temp_relation_type_dct[temp_relation_id])
    #             if len(temp_relation_type_dct[temp_relation_id]) > max_type_num_per_relation:
    #                 max_type_num_per_relation = len(temp_relation_type_dct[temp_relation_id])
    #         else:
    #             print(k)
    #             relation_type_dct[k] = []
    #
    #             # print(len(self.entity_ids))
    #
    #     print(max_type_num_per_relation)
    #     print(max_relation_type_id)
    #     # print(relation_type_dct)
    #     return relation_type_dct

    def get_parameters(self):

        parameters = {'entity_embedding_table':self.entity_embedding_table,
                      'relation_embedding_table':self.relation_embedding_table,
                      'embedding_table':self.embedding_table,
                      'batch_size':self.batch_size, 'max_path_mun':self.max_path_mun,
                      'train_samples':self.train_samples, 'train_samples_num':self.train_samples_num,
                      'batch_num_train':self.batch_num_train, 'sample_ids_train':self.sample_ids_train,
                      'max_train_path_len':self.max_train_path_len,
                      'test_samples': self.test_samples, 'test_samples_num': self.test_samples_num,
                      'batch_num_test': self.batch_num_test, 'sample_ids_test': self.sample_ids_test,
                      'max_test_path_len': self.max_test_path_len,
                      'entity_type_num': self.entity_type_num
                      }

        return parameters

    def relationID2taskID(self):

        if self.dataset_name == 'NELL-995':
            if self.task == 'all':
                task_list = os.listdir('{}/{}'.format(self.base_path, 'tasks'))
            else:
                task_list = [self.task]
            relationIDs = []
            task_id = 0
            taskIDs = []
            for task in task_list:
                relationID = int(self.relation_ids[':'.join(task.split('_'))]) - self.entity_num
                relationIDs.append(relationID)
                taskIDs.append(task_id)
                task_id += 1
            # print(relationID)

        elif self.dataset_name == 'FB15k-237':
            if self.task == 'all':
                task_list = os.listdir('{}/{}'.format(self.base_path, 'tasks'))
            else:
                task_list = [self.task]
            relationIDs = []
            task_id = 0
            taskIDs = []
            for task in task_list:
                task = '/' + '/'.join(task.split('@'))
                # print(task)
                relationID = int(self.relation_ids[task]) - self.entity_num
                relationIDs.append(relationID)
                taskIDs.append(task_id)
                task_id += 1
            # print(relationID)

        return (relationIDs, taskIDs)

    def entity2id(self):
        entityID = {}
        file_path = '{}/{}'.format(self.base_path, 'entity2id.txt')

        if self.dataset_name == 'NELL-995':
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    entity, id = line.split('\n')[0].split('\t')
                    # print(entity, id)
                    entityID[entity] = int(id)
        elif self.dataset_name =='FB15k-237':
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    entity_id = line.split('\n')[0].split(' ')
                    entity = entity_id[0]
                    id = entity_id[-1]
                    # entity, id = line.split('\n')[0].split('\t')
                    entityID[entity] = int(id)
                    # print(entity_id)

        return entityID, len(entityID)

    def relation2id(self):
        relationID = {}
        file_path = '{}/{}'.format(self.base_path, 'relation2id.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                relation, id = line.split('\n')[0].split('\t')
                # print(entity, id)
                relationID[relation] = int(id) + self.entity_num
        return relationID, len(relationID)

    def read_single_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def create_train_sample_dct(self):
        if self.task == 'all':
            task_list = os.listdir('{}/{}'.format(self.base_path, 'tasks'))
        else:
            task_list = [self.task]
        task2id = {}
        org_enti_pair_dct = {}
        pos_samples = {}
        org_enti_pair_count = 0
        sample_count = 0

        # for i in range(len(task_list)):
        #     task2id[task_list[i].split('_')[-1]] = i
        if self.task == 'all':
            for task in task_list:
                file_path = '{}/{}/{}/{}_{}.json'.format(self.base_path, 'tasks', task, 'sample_paths_train', str(self.max_path_mun))
                json_data = self.read_single_json(file_path)
                for k, v in json_data.items():  # 对每个entity pair
                    if v['label'] == 1 and len(v['sample_paths']) != 0:
                        paths = v['sample_paths']  # [[[e1,e2,e3,...], [r1,r2,...]], [[], []],...]
                        pathID_list = []
                        pathID_str_list = []
                        for path in paths:  # [[e1,e2,e3,...], [r1,r2,...]]
                            entity_set, relation_set = path
                            single_path_id = []
                            single_path_str = ''
                            for i in range(len(relation_set)):
                                entity_id = self.entity_ids[entity_set[i]]
                                relation_id = self.relation_ids[relation_set[i]]

                                single_path_id.append(entity_id)
                                single_path_id.append(relation_id)

                                single_path_str += '{}-{}-'.format(str(entity_id), str(relation_id))

                            single_path_id.append(self.entity_ids[entity_set[-1]])  # [e1,r1,e2,r2,e3,...]
                            single_path_str += str(self.entity_ids[entity_set[-1]])  # e1-r1-e2-r2-e3-...
                            # print(v["origin entity pair"], single_path_id)
                            if single_path_str not in pathID_str_list:
                                pathID_str_list.append(single_path_str)
                                pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                            if len(single_path_id) > self.max_train_path_len:
                                self.max_train_path_len = len(single_path_id)
                        # s = ':'
                        pos_samples[sample_count] = {'origin_entity_pair': [pathID_list[0][0], pathID_list[0][-1]],
                                                     'sample_paths': pathID_list,
                                                     'task_id': self.relation_ids[':'.join(task.split('_'))]}
                        sample_count += 1
        elif self.task in os.listdir('{}/{}'.format(self.base_path, 'tasks')):
            file_path = '{}/{}/{}/{}_{}.json'.format(self.base_path, 'tasks', self.task, 'sample_paths_train', str(self.max_path_mun))
            json_data = self.read_single_json(file_path)
            for k, v in json_data.items():
                if len(v['sample_paths']) != 0 and v["origin entity pair"][0] != v["origin entity pair"][1]:
                    paths = v['sample_paths']  # [[[e1,e2,e3,...], [r1,r2,...]], [[], []],...]
                    pathID_list = []
                    pathID_str_list = []
                    entity_type_ids = []
                    entity_flags_set = []
                    if len(paths) > self.split_path_num:
                        paths = paths[:self.split_path_num]
                    for t in range(len(paths)):  # [[e1,e2,e3,...], [r1,r2,...]]
                        entity_set, relation_set = paths[t]
                        single_path_id = []
                        single_path_str = ''

                        single_path_entity_type_ids = []
                        # single_path_entity_flag = []
                        # entity_flag_in_path = 0

                        for i in range(len(relation_set)):
                            entity_id = self.entity_ids[entity_set[i]]

                            per_entity_type_ids = self.entity_type_dct[entity_set[i]]
                            if len(per_entity_type_ids) < self.max_type_num_per_entity:
                                per_entity_type_ids += [self.entity_type_num] * int(self.max_type_num_per_entity - len(per_entity_type_ids))
                            # print(per_entity_type_ids)
                            single_path_entity_type_ids.append([per_entity_type_ids])

                            relation_id = self.relation_ids[relation_set[i]]

                            single_path_id.append(entity_id)
                            single_path_id.append(relation_id)

                            single_path_str += '{}-{}-'.format(str(entity_id), str(relation_id))

                        single_path_id.append(self.entity_ids[entity_set[-1]])  # [e1,r1,e2,r2,e3,...]
                        single_path_str += str(self.entity_ids[entity_set[-1]])  # e1-r1-e2-r2-e3-...

                        last_entity_type_ids = self.entity_type_dct[entity_set[-1]]
                        if len(last_entity_type_ids) < self.max_type_num_per_entity:
                            last_entity_type_ids += [self.entity_type_num] * int(self.max_type_num_per_entity - len(last_entity_type_ids))
                        single_path_entity_type_ids.append([last_entity_type_ids])
                        single_path_entity_type_ids = np.concatenate(single_path_entity_type_ids, axis=0)

                        entity_flag_in_single_path = np.array(range(len(entity_set))) * 2
                        # print(entity_set)
                        # print(entity_flag_in_single_path)
                        # print(single_path_id)
                        # print(single_path_entity_type_ids)
                        # print('.................')

                        # print(v["origin entity pair"], single_path_id)
                        if single_path_str not in pathID_str_list:
                            pathID_str_list.append(single_path_str)
                            pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                            entity_type_ids.append(single_path_entity_type_ids)
                            entity_flags_set.append(entity_flag_in_single_path.tolist())

                        # pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                        # entity_type_ids.append(single_path_entity_type_ids)
                        # entity_flags_set.append(entity_flag_in_single_path.tolist())
                        if len(single_path_id) > self.max_train_path_len:
                            self.max_train_path_len = len(single_path_id)
                    # s = ':'
                    pos_samples[sample_count] = {'origin_entity_pair': [pathID_list[0][0], pathID_list[0][-1]],
                                                 'sample_paths': pathID_list,
                                                 'entity_type_ids': entity_type_ids,
                                                 'entity_flags_set': entity_flags_set,
                                                 'task_id': v['label']} # self.relation_ids[':'.join(self.task.split('_'))]
                    sample_count += 1
        # print(self.max_train_path_len)

        return pos_samples

    def create_test_sample_dct(self):
        if self.task == 'all':
            task_list = os.listdir('{}/{}'.format(self.base_path, 'tasks'))
        else:
            task_list = [self.task]
        task2id = {}
        org_enti_pair_dct = {}
        pos_samples = {}
        org_enti_pair_count = 0
        sample_count = 0

        # for i in range(len(task_list)):
        #     task2id[task_list[i].split('_')[-1]] = i
        if self.task == 'all':
            for task in task_list:
                file_path = '{}/{}/{}/{}_{}.json'.format(self.base_path, 'tasks', task, 'sample_paths_test', str(self.max_path_mun))
                json_data = self.read_single_json(file_path)
                for k, v in json_data.items():  # 对每个entity pair
                    if v['label'] == 1 and len(v['sample_paths']) != 0:
                        paths = v['sample_paths']  # [[[e1,e2,e3,...], [r1,r2,...]], [[], []],...]
                        pathID_list = []
                        pathID_str_list = []
                        for path in paths:  # [[e1,e2,e3,...], [r1,r2,...]]
                            entity_set, relation_set = path
                            single_path_id = []
                            single_path_str = ''
                            for i in range(len(relation_set)):
                                entity_id = self.entity_ids[entity_set[i]]
                                relation_id = self.relation_ids[relation_set[i]]

                                single_path_id.append(entity_id)
                                single_path_id.append(relation_id)

                                single_path_str += '{}-{}-'.format(str(entity_id), str(relation_id))

                            single_path_id.append(self.entity_ids[entity_set[-1]])  # [e1,r1,e2,r2,e3,...]
                            single_path_str += str(self.entity_ids[entity_set[-1]])  # e1-r1-e2-r2-e3-...
                            # print(v["origin entity pair"], single_path_id)
                            if single_path_str not in pathID_str_list:
                                pathID_str_list.append(single_path_str)
                                pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                            if len(single_path_id) > self.max_test_path_len:
                                self.max_test_path_len = len(single_path_id)
                        # s = ':'
                        pos_samples[sample_count] = {'origin_entity_pair': [pathID_list[0][0], pathID_list[0][-1]],
                                                     'sample_paths': pathID_list,
                                                     'task_id': self.relation_ids[':'.join(task.split('_'))]}
                        sample_count += 1
        elif self.task in os.listdir('{}/{}'.format(self.base_path, 'tasks')):
            # file_path = '{}/{}/{}/{}_{}.json'.format(self.base_path, 'tasks', self.task, 'sample_paths_test', str(self.max_path_mun))
            file_path = '{}/{}/{}/{}_{}.json'.format(self.base_path, 'tasks', self.task, 'sample_paths_test', 'demo')
            json_data = self.read_single_json(file_path)
            for k, v in json_data.items():
                if len(v['sample_paths']) != 0 and v["origin entity pair"][0] != v["origin entity pair"][1]:
                    paths = v['sample_paths']  # [[[e1,e2,e3,...], [r1,r2,...]], [[], []],...]
                    pathID_list = []
                    pathID_str_list = []
                    entity_type_ids = []
                    entity_flags_set = []
                    if len(paths) > self.split_path_num:
                        paths = paths[:self.split_path_num]
                    for path in paths:  # [[e1,e2,e3,...], [r1,r2,...]]
                        entity_set, relation_set = path
                        single_path_id = []
                        single_path_str = ''

                        single_path_entity_type_ids = []
                        # single_path_entity_flag = []
                        # entity_flag_in_path = 0

                        for i in range(len(relation_set)):
                            entity_id = self.entity_ids[entity_set[i]]

                            per_entity_type_ids = self.entity_type_dct[entity_set[i]]
                            if len(per_entity_type_ids) < self.max_type_num_per_entity:
                                per_entity_type_ids += [self.entity_type_num] * int(self.max_type_num_per_entity - len(per_entity_type_ids))
                            # print(per_entity_type_ids)
                            single_path_entity_type_ids.append([per_entity_type_ids])

                            relation_id = self.relation_ids[relation_set[i]]

                            single_path_id.append(entity_id)
                            single_path_id.append(relation_id)

                            single_path_str += '{}-{}-'.format(str(entity_id), str(relation_id))

                        single_path_id.append(self.entity_ids[entity_set[-1]])  # [e1,r1,e2,r2,e3,...]
                        single_path_str += str(self.entity_ids[entity_set[-1]])  # e1-r1-e2-r2-e3-...

                        last_entity_type_ids = self.entity_type_dct[entity_set[-1]]
                        if len(last_entity_type_ids) < self.max_type_num_per_entity:
                            last_entity_type_ids += [self.entity_type_num] * int(self.max_type_num_per_entity - len(last_entity_type_ids))
                        single_path_entity_type_ids.append([last_entity_type_ids])
                        single_path_entity_type_ids = np.concatenate(single_path_entity_type_ids, axis=0)

                        entity_flag_in_single_path = np.array(range(len(entity_set))) * 2
                        # print(entity_set)
                        # print(entity_flag_in_single_path)
                        # print(single_path_id)
                        # print(single_path_entity_type_ids)
                        # print('.................')

                        # print(v["origin entity pair"], single_path_id)

                        if single_path_str not in pathID_str_list:
                            pathID_str_list.append(single_path_str)
                            pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                            entity_type_ids.append(single_path_entity_type_ids)
                            entity_flags_set.append(entity_flag_in_single_path.tolist())
                        # pathID_list.append(single_path_id)  # [[e1,r1,e2,r2,e3,...], [],...]
                        # entity_type_ids.append(single_path_entity_type_ids)
                        # entity_flags_set.append(entity_flag_in_single_path.tolist())
                        if len(single_path_id) > self.max_test_path_len:
                            self.max_test_path_len = len(single_path_id)
                    # s = ':'
                    pos_samples[sample_count] = {'origin_entity_pair': [pathID_list[0][0], pathID_list[0][-1]],
                                                 'sample_paths': pathID_list,
                                                 'entity_type_ids': entity_type_ids,
                                                 'entity_flags_set': entity_flags_set,
                                                 'task_id': v['label']} # self.relation_ids[':'.join(self.task.split('_'))]
                    sample_count += 1

        return pos_samples

# read_vec()
# dataloader = DataLoader(dataset_name='NELL-995', batch_size=4, shuffle=True)
# dataloader.load()

# for k in range(dataloader.dataIterator = dataloader.load()batch_num):

    # print(k)
    # print(type(dataloader.load()))
    # dataloader.load()
    # i, batch_sample_features, batch_sample_label = next(dataIterator)
    # print(i, batch_sample_features.shape, batch_sample_label)
# print(dataloader.embedding_table.shape)
# print(dataloader.batch_num)