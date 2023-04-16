from dataloader import DataLoader
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, roc_curve, auc
import os
import random
import warnings
import sys
from collections import defaultdict

warnings.filterwarnings("ignore")
GPUs = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(GPUs[0], True)

def seed_(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_(seed=42)

class Attention(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, attention_head, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention_head = attention_head
        self.embedding_size = embedding_size
        # self.Q = tf.Variable(tf.random.normal((feature_size, embedding_size), mean=0, stddev=0.01))
        # self.K = tf.Variable(tf.random.normal((feature_size, embedding_size), mean=0, stddev=0.01))
        # self.V = tf.Variable(tf.random.normal((feature_size, embedding_size), mean=0, stddev=0.01))
        self.Q = tf.keras.layers.Dense(embedding_size, activation=None, use_bias=True)
        self.K = tf.keras.layers.Dense(embedding_size, activation=None, use_bias=True)
        self.V = tf.keras.layers.Dense(embedding_size, activation=None, use_bias=True)
        # self.path_weight_mat = tf.keras.layers.Dense(1)
        # self.fc = tf.keras.layers.Dense(feature_size, activation=tf.nn.relu, use_bias=True)
        self.Dropout = tf.keras.layers.Dropout(drop_rate)

        # self.layerNormal_1 = tf.keras.layers.LayerNormalization(center=False, scale=False)
        # self.layerNormal_2 = tf.keras.layers.LayerNormalization(center=False, scale=False)

    def call(self, input, entityType_embedding, mask, training=None):
        input = self.Dropout(input, training=training)
        # print(input.shape)
        # max_path_len = input.shape[-2]
        # entity_flag_per_path = list(range(0, max_path_len, 2))

        # print(entity_flag_per_path)
        # batch x max_path_num x max_path_len x embedding_size
        # query = tf.matmul(input, self.Q)
        # key = tf.matmul(input, self.K)
        # value = tf.matmul(input, self.V)
        query = self.Q(input) + entityType_embedding #+ entityType_embedding #tf.matmul(input, self.Q)
        key = self.K(input) #tf.matmul(input, self.K)
        value = self.V(input) #tf.matmul(input, self.V)
        # print(tf.expand_dims(query, axis=1))
        query_MH = tf.concat(tf.split(tf.expand_dims(query, axis=1), self.attention_head, axis=-1), axis=1)
        key_MH = tf.concat(tf.split(tf.expand_dims(key, axis=1), self.attention_head, axis=-1), axis=1)
        value_MH = tf.concat(tf.split(tf.expand_dims(value, axis=1), self.attention_head, axis=-1), axis=1)

        # batch x head x max_path_num x max_path_len x max_path_len
        # atten_score = tf.matmul(query, key, transpose_b=True)
        atten_score = tf.matmul(query_MH, key_MH, transpose_b=True)/tf.math.sqrt(self.embedding_size/self.attention_head)
        inf_flag = tf.ones_like(atten_score) * -1e10
        zero_flag = tf.zeros_like(atten_score)
        atten_mat = tf.where(np.expand_dims(np.expand_dims(mask, axis=-2), axis=1) == 1, x=atten_score, y=inf_flag)
        atten_mat = tf.nn.softmax(atten_mat, axis=-1)
        atten_mat = tf.where(np.expand_dims(np.expand_dims(mask, axis=-1), axis=1) == 1, x=atten_mat, y=zero_flag)
        atten_mat = self.Dropout(atten_mat, training=training)

        h = tf.matmul(atten_mat, value_MH)

        # batch x max_path_num x max_path_len x embedding_size
        e_r_embedding = tf.squeeze(tf.concat(tf.split(h, self.attention_head, axis=1), axis=-1), axis=1)
        # e_r_embedding = self.layerNormal_1(e_r_embedding)

        # batch x max_path_num x max_path_len x feature_size
        # e_r_embedding = self.fc(e_r_embedding) + input
        # e_r_embedding = self.layerNormal_2(e_r_embedding)

        return e_r_embedding #e_r_embedding

class Model(tf.keras.Model):
    def __init__(self, dataset, feature_size, embedding_size, attention_head, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.dataset_parameters = dataset.get_parameters()
        self.entity_num = self.dataset_parameters['entity_embedding_table'].shape[0]
        self.relation_num = self.dataset_parameters['relation_embedding_table'].shape[0]
        self.entity_type_num = self.dataset_parameters['entity_type_num']
        # self.entity_embedding_table = tf.Variable(initial_value=self.dataset_parameters['entity_embedding_table'],
        #                                           trainable=True,
        #                                           dtype=tf.float32)
        # self.relation_embedding_table = tf.Variable(initial_value=self.dataset_parameters['relation_embedding_table'],
        #                                             trainable=True,
        #                                             dtype=tf.float32)
        self.embedding_table = tf.Variable(initial_value=self.dataset_parameters['embedding_table'], #self.dataset_parameters['embedding_table'],tf.random.normal((self.entity_num + self.relation_num, feature_size), mean=0, stddev=0.01)
                                           trainable=True,
                                           dtype=tf.float32)
        # self.entity_embedding_table = tf.Variable(tf.random.normal((self.entity_num, feature_size), mean=0, stddev=0.01), dtype=tf.float32)
        # self.relation_embedding_table = tf.Variable(tf.random.normal((self.relation_num, feature_size), mean=0, stddev=0.01), dtype=tf.float32)
        # self.embedding_table = tf.concat([self.entity_embedding_table, self.relation_embedding_table], axis=0)
        self.entity_type_embedding_table = tf.Variable(initial_value=tf.concat(
                                                                               [tf.random.normal(
                                                                                   (self.entity_type_num, embedding_size), mean=0, stddev=0.01
                                                                               ),
                                                                                   tf.zeros((1, embedding_size))
                                                                               ], axis=0),
                                                       trainable=True,
                                                       dtype=tf.float32)

        self.batch_size = self.dataset_parameters['batch_size']
        self.max_path_mun = self.dataset_parameters['max_path_mun']

        self.batch_num_train = self.dataset_parameters['batch_num_train']
        self.train_samples = self.dataset_parameters['train_samples']
        self.sample_ids_train = self.dataset_parameters['sample_ids_train']
        self.max_train_path_len = self.dataset_parameters['max_train_path_len']
        self.batch_num_test = self.dataset_parameters['batch_num_test']
        self.test_samples = self.dataset_parameters['test_samples']
        self.sample_ids_test = self.dataset_parameters['sample_ids_test']
        self.max_test_path_len = self.dataset_parameters['max_test_path_len']

        self.attention_layer1 = Attention(feature_size, embedding_size, attention_head, drop_rate)
        # self.attention_layer2 = Attention(feature_size, embedding_size, attention_head, drop_rate)
        self.attention_layer3 = Attention(feature_size, embedding_size, attention_head, drop_rate)

        self.path_weight_mat1 = tf.keras.layers.Dense(1, activation=None, use_bias=True)
        self.agg_fc1 = tf.keras.layers.Dense(feature_size, activation=tf.nn.relu, use_bias=True)
        self.path_weight_mat2 = tf.keras.layers.Dense(1, activation=None, use_bias=True)
        self.agg_fc2 = tf.keras.layers.Dense(feature_size, activation=tf.nn.relu, use_bias=True)

        # self.agg_fc_2 = tf.keras.layers.Dense(feature_size, activation=tf.nn.relu, use_bias=True)

        # self.entityType_fc = tf.keras.layers.Dense(embedding_size, activation=tf.nn.relu)
        self.entityType_weight_mat = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def train_loader(self):
        for i in range(self.batch_num_train):
            # batch_sample_ids = self.sample_ids_train[i*self.batch_size:(i+1)*self.batch_size]
            batch_sample_features = []
            batch_mask = []
            batch_sample_label = []
            batch_origin_entity_pair_ids = []
            # batch_path_entity_flags_set = []
            batch_entity_type_ids = []

            batch_origin_entity_pair_ids_in_path = []

            for sample_id in self.sample_ids_train[i*self.batch_size:(i+1)*self.batch_size]:

                single_pos_sample = self.train_samples[sample_id]
                # print(single_pos_sample['entity_type_ids'])
                # print(single_pos_sample['entity_flags_set'])

                batch_sample_label.append(single_pos_sample['task_id'])
                single_pos_sample_path_features = []
                single_pos_sample_path_len_mask = []

                # single_pos_sample_path_entity_flags_set = []
                single_pos_sample_path_entity_type_ids = []

                single_pos_sample_path_origin_entity_pair_ids = []

                batch_origin_entity_pair_ids.append(single_pos_sample['origin_entity_pair'])

                for i in range(len(single_pos_sample['sample_paths'])):
                    path = single_pos_sample['sample_paths'][i] # [e1,r1,e2,r2,e3,...]
                    if len(path) < 2:
                        continue

                    # entity_flags_set = single_pos_sample['entity_flags_set'][i]
                    entity_type_ids = single_pos_sample['entity_type_ids'][i]
                    max_entity_num = int(self.max_train_path_len/2) + 1
                    # print(self.max_train_path_len)
                    # if entity_type_ids.shape[0] < max_entity_num:
                    #     entity_type_ids = tf.concat([entity_type_ids,
                    #                                  tf.ones(
                    #                                      (max_entity_num - entity_type_ids.shape[0],
                    #                                       entity_type_ids.shape[1]),
                    #                                    dtype=tf.int32
                    #                                ) * self.entity_type_num
                    #                                ],
                    #                               axis=0
                    #                               ) # max_entity_num x max_type_num_per_entity
                    entity_type_ids = tf.concat([entity_type_ids,
                                                 tf.ones(
                                                     (max_entity_num - entity_type_ids.shape[0],
                                                      entity_type_ids.shape[1]),
                                                     dtype=tf.int32
                                                 ) * self.entity_type_num
                                                 ],
                                                axis=0
                                                )  # max_entity_num x max_type_num_per_entity
                    # entity_flags_set += [-1] * (max_entity_num - len(entity_flags_set))
                    # print(entity_type_ids)
                    # print(entity_flags_set)

                    e_r_embedding = tf.gather(self.embedding_table, path)

                    # origin_entity_pair_ids_in_path = [[1] + [0] * int(len(path)-2) + [1]]

                    if e_r_embedding.shape[0] < self.max_train_path_len: # 39
                        path_len_mask = [[1] * e_r_embedding.shape[0] + [0] * (self.max_train_path_len - e_r_embedding.shape[0])]
                        origin_entity_pair_ids_in_path = [[-1] + [0] * int(e_r_embedding.shape[0] - 2) + [1] + [0] * (self.max_train_path_len - e_r_embedding.shape[0])]
                        # print(len(origin_entity_pair_ids_in_path[0]), e_r_embedding.shape[0], self.max_train_path_len, path)
                        # print(origin_entity_pair_ids_in_path)
                        e_r_embedding = tf.concat([e_r_embedding,
                                                   tf.zeros(
                                                       (self.max_train_path_len - e_r_embedding.shape[0],
                                                        e_r_embedding.shape[1]),
                                                       dtype=tf.float32
                                                   )
                                                   ],
                                                  axis=0
                                                  ) # max_path_len x feature_size
                    else:
                        path_len_mask = [[1] * e_r_embedding.shape[0]]
                        origin_entity_pair_ids_in_path = [[-1] + [0] * int(self.max_train_path_len - 2) + [1]]
                    single_pos_sample_path_len_mask.append(path_len_mask)
                    single_pos_sample_path_features.append(tf.expand_dims(e_r_embedding, axis=0))

                    # single_pos_sample_path_entity_flags_set.append([entity_flags_set])
                    single_pos_sample_path_entity_type_ids.append(tf.expand_dims(entity_type_ids, axis=0))

                    single_pos_sample_path_origin_entity_pair_ids.append(origin_entity_pair_ids_in_path)

                single_pos_sample_path_len_mask = np.concatenate(single_pos_sample_path_len_mask, axis=0)# path_mun x max_path_len
                single_pos_sample_path_features = tf.concat(single_pos_sample_path_features, axis=0)# path_mun x max_path_len x feature_size
                single_pos_sample_path_origin_entity_pair_ids = np.concatenate(single_pos_sample_path_origin_entity_pair_ids, axis=0)# path_mun x max_path_len

                # single_pos_sample_path_entity_flags_set = np.concatenate(single_pos_sample_path_entity_flags_set, axis=0)# path_mun x max_entity_num
                # print(single_pos_sample_path_entity_type_ids)
                single_pos_sample_path_entity_type_ids = np.concatenate(single_pos_sample_path_entity_type_ids, axis=0)# path_mun x max_entity_num x max_type_num_per_entity

                if single_pos_sample_path_features.shape[0] < self.max_path_mun: # 5

                    # single_pos_sample_path_entity_flags_set = np.concatenate([single_pos_sample_path_entity_flags_set,
                    #                                                           np.ones((self.max_path_mun -
                    #                                                                    single_pos_sample_path_features.shape[
                    #                                                                        0],
                    #                                                                    int(self.max_train_path_len / 2) + 1),
                    #                                                                   dtype=np.int32
                    #                                                                   ) * (-1) # self.max_train_path_len - 1
                    #                                                           ],
                    #                                                          axis=0
                    #                                                          )  # max_path_mun x max_entity_num

                    single_pos_sample_path_origin_entity_pair_ids = np.concatenate([single_pos_sample_path_origin_entity_pair_ids,
                                                                                    np.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                              self.max_train_path_len
                                                                                              ),
                                                                                            dtype=np.int32)
                                                                                    ],
                                                                                   axis=0
                                                                                   )# max_path_mun x max_path_len

                    single_pos_sample_path_entity_type_ids = tf.concat([single_pos_sample_path_entity_type_ids,
                                                                        tf.ones((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                 int(self.max_train_path_len/2) + 1,
                                                                                 single_pos_sample_path_entity_type_ids.shape[-1]),
                                                                                dtype=tf.int32
                                                                                ) * self.entity_type_num
                                                                        ],
                                                                       axis=0
                                                                       )# max_path_mun x max_entity_num x max_type_num_per_entity

                    single_pos_sample_path_len_mask = np.concatenate([single_pos_sample_path_len_mask,
                                                                      np.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                self.max_train_path_len)
                                                                               )
                                                                      ],
                                                                     axis=0
                                                                     )

                    single_pos_sample_path_features = tf.concat([single_pos_sample_path_features,
                                                                 tf.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                           self.max_train_path_len,
                                                                           self.embedding_table.shape[1]),
                                                                          dtype=tf.float32
                                                                          )
                                                                 ],
                                                                axis=0
                                                                )# max_path_mun x max_path_len x feature_size


                # print(single_pos_sample_path_entity_type_ids)
                batch_origin_entity_pair_ids_in_path.append(np.expand_dims(single_pos_sample_path_origin_entity_pair_ids, axis=0))
                batch_mask.append(np.expand_dims(single_pos_sample_path_len_mask, axis=0))
                batch_sample_features.append(tf.expand_dims(single_pos_sample_path_features, axis=0))

                # batch_path_entity_flags_set.append(np.expand_dims(single_pos_sample_path_entity_flags_set, axis=0))
                batch_entity_type_ids.append(tf.expand_dims(single_pos_sample_path_entity_type_ids, axis=0))

            batch_origin_entity_pair_ids_in_path = np.concatenate(batch_origin_entity_pair_ids_in_path, axis=0) # batch x max_path_mun x max_path_len
            batch_mask = np.concatenate(batch_mask, axis=0)# batch_size x max_path_mun x max_path_len
            batch_sample_features = tf.concat(batch_sample_features, axis=0)# batch_size x max_path_mun x max_path_len x feature_size
            # batch_path_entity_flags_set = np.concatenate(batch_path_entity_flags_set, axis=0)# batch_size x max_path_mun x max_entity_num
            batch_entity_type_ids = tf.concat(batch_entity_type_ids, axis=0)# batch_size x max_path_mun x max_entity_num x max_type_num_per_entity
            yield batch_sample_features, batch_mask, batch_origin_entity_pair_ids, \
                  batch_origin_entity_pair_ids_in_path, np.array(batch_sample_label), batch_entity_type_ids

    def test_loader(self):
        for i in range(self.batch_num_test):
            # batch_sample_ids = self.sample_ids_test[i*self.batch_size:(i+1)*self.batch_size]
            batch_sample_features = []
            batch_mask = []
            batch_sample_label = []
            batch_origin_entity_pair_ids = []
            # batch_path_entity_flags_set = []
            batch_entity_type_ids = []

            batch_origin_entity_pair_ids_in_path = []

            for sample_id in self.sample_ids_test[i*self.batch_size:(i+1)*self.batch_size]:

                single_pos_sample = self.test_samples[sample_id]
                # print(single_pos_sample['entity_type_ids'])
                # print(single_pos_sample['entity_flags_set'])

                batch_sample_label.append(single_pos_sample['task_id'])
                single_pos_sample_path_features = []
                single_pos_sample_path_len_mask = []

                # single_pos_sample_path_entity_flags_set = []
                single_pos_sample_path_entity_type_ids = []

                single_pos_sample_path_origin_entity_pair_ids = []

                batch_origin_entity_pair_ids.append(single_pos_sample['origin_entity_pair'])

                for i in range(len(single_pos_sample['sample_paths'])):
                    path = single_pos_sample['sample_paths'][i] # [e1,r1,e2,r2,e3,...]

                    # entity_flags_set = single_pos_sample['entity_flags_set'][i]
                    entity_type_ids = single_pos_sample['entity_type_ids'][i]
                    max_entity_num = int(self.max_test_path_len/2) + 1
                    # print(self.max_test_path_len)
                    # if entity_type_ids.shape[0] < max_entity_num:
                    #     entity_type_ids = tf.concat([entity_type_ids,
                    #                                  tf.ones(
                    #                                      (max_entity_num - entity_type_ids.shape[0],
                    #                                       entity_type_ids.shape[1]),
                    #                                    dtype=tf.int32
                    #                                ) * self.entity_type_num
                    #                                ],
                    #                               axis=0
                    #                               ) # max_entity_num x max_type_num_per_entity
                    entity_type_ids = tf.concat([entity_type_ids,
                                                 tf.ones(
                                                     (max_entity_num - entity_type_ids.shape[0],
                                                      entity_type_ids.shape[1]),
                                                     dtype=tf.int32
                                                 ) * self.entity_type_num
                                                 ],
                                                axis=0
                                                )  # max_entity_num x max_type_num_per_entity
                    # entity_flags_set += [-1] * (max_entity_num - len(entity_flags_set))
                    # print(entity_type_ids)
                    # print(entity_flags_set)

                    e_r_embedding = tf.gather(self.embedding_table, path)

                    # origin_entity_pair_ids_in_path = [[1] + [0] * int(len(path)-2) + [1]]

                    if e_r_embedding.shape[0] < self.max_test_path_len: # 39
                        path_len_mask = [[1] * e_r_embedding.shape[0] + [0] * (self.max_test_path_len - e_r_embedding.shape[0])]
                        origin_entity_pair_ids_in_path = [[-1] + [0] * int(len(path) - 2) + [1] + [0] * (self.max_test_path_len - e_r_embedding.shape[0])]
                        e_r_embedding = tf.concat([e_r_embedding,
                                                   tf.zeros(
                                                       (self.max_test_path_len - e_r_embedding.shape[0],
                                                        e_r_embedding.shape[1]),
                                                       dtype=tf.float32
                                                   )
                                                   ],
                                                  axis=0
                                                  ) # max_path_len x feature_size
                    else:
                        path_len_mask = [[1] * e_r_embedding.shape[0]]
                        origin_entity_pair_ids_in_path = [[-1] + [0] * int(len(path) - 2) + [1]]
                    single_pos_sample_path_len_mask.append(path_len_mask)
                    single_pos_sample_path_features.append(tf.expand_dims(e_r_embedding, axis=0))

                    # single_pos_sample_path_entity_flags_set.append([entity_flags_set])
                    single_pos_sample_path_entity_type_ids.append(tf.expand_dims(entity_type_ids, axis=0))

                    single_pos_sample_path_origin_entity_pair_ids.append(origin_entity_pair_ids_in_path)

                single_pos_sample_path_len_mask = np.concatenate(single_pos_sample_path_len_mask, axis=0)# path_mun x max_path_len
                single_pos_sample_path_features = tf.concat(single_pos_sample_path_features, axis=0)# path_mun x max_path_len x feature_size
                single_pos_sample_path_origin_entity_pair_ids = np.concatenate(single_pos_sample_path_origin_entity_pair_ids, axis=0)# path_mun x max_path_len

                # single_pos_sample_path_entity_flags_set = np.concatenate(single_pos_sample_path_entity_flags_set, axis=0)# path_mun x max_entity_num
                # print(single_pos_sample_path_entity_type_ids)
                single_pos_sample_path_entity_type_ids = np.concatenate(single_pos_sample_path_entity_type_ids, axis=0)# path_mun x max_entity_num x max_type_num_per_entity

                if single_pos_sample_path_features.shape[0] < self.max_path_mun: # 5

                    # single_pos_sample_path_entity_flags_set = np.concatenate([single_pos_sample_path_entity_flags_set,
                    #                                                           np.ones((self.max_path_mun -
                    #                                                                    single_pos_sample_path_features.shape[
                    #                                                                        0],
                    #                                                                    int(self.max_test_path_len / 2) + 1),
                    #                                                                   dtype=np.int32
                    #                                                                   ) * (-1) # self.max_test_path_len - 1
                    #                                                           ],
                    #                                                          axis=0
                    #                                                          )  # max_path_mun x max_entity_num

                    single_pos_sample_path_origin_entity_pair_ids = np.concatenate([single_pos_sample_path_origin_entity_pair_ids,
                                                                                    np.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                              self.max_test_path_len
                                                                                              ),
                                                                                            dtype=np.int32)
                                                                                    ],
                                                                                   axis=0
                                                                                   )# max_path_mun x max_path_len

                    single_pos_sample_path_entity_type_ids = tf.concat([single_pos_sample_path_entity_type_ids,
                                                                        tf.ones((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                 int(self.max_test_path_len/2) + 1,
                                                                                 single_pos_sample_path_entity_type_ids.shape[-1]),
                                                                                dtype=tf.int32
                                                                                ) * self.entity_type_num
                                                                        ],
                                                                       axis=0
                                                                       )# max_path_mun x max_entity_num x max_type_num_per_entity

                    single_pos_sample_path_len_mask = np.concatenate([single_pos_sample_path_len_mask,
                                                                      np.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                                self.max_test_path_len)
                                                                               )
                                                                      ],
                                                                     axis=0
                                                                     )

                    single_pos_sample_path_features = tf.concat([single_pos_sample_path_features,
                                                                 tf.zeros((self.max_path_mun - single_pos_sample_path_features.shape[0],
                                                                           self.max_test_path_len,
                                                                           self.embedding_table.shape[1]),
                                                                          dtype=tf.float32
                                                                          )
                                                                 ],
                                                                axis=0
                                                                )# max_path_mun x max_path_len x feature_size


                # print(single_pos_sample_path_entity_type_ids)
                batch_origin_entity_pair_ids_in_path.append(np.expand_dims(single_pos_sample_path_origin_entity_pair_ids, axis=0))
                batch_mask.append(np.expand_dims(single_pos_sample_path_len_mask, axis=0))
                batch_sample_features.append(tf.expand_dims(single_pos_sample_path_features, axis=0))

                # batch_path_entity_flags_set.append(np.expand_dims(single_pos_sample_path_entity_flags_set, axis=0))
                batch_entity_type_ids.append(tf.expand_dims(single_pos_sample_path_entity_type_ids, axis=0))

            batch_origin_entity_pair_ids_in_path = np.concatenate(batch_origin_entity_pair_ids_in_path, axis=0) # batch x max_path_mun x max_path_len
            batch_mask = np.concatenate(batch_mask, axis=0)# batch_size x max_path_mun x max_path_len
            batch_sample_features = tf.concat(batch_sample_features, axis=0)# batch_size x max_path_mun x max_path_len x feature_size
            # batch_path_entity_flags_set = np.concatenate(batch_path_entity_flags_set, axis=0)# batch_size x max_path_mun x max_entity_num
            batch_entity_type_ids = tf.concat(batch_entity_type_ids, axis=0)# batch_size x max_path_mun x max_entity_num x max_type_num_per_entity
            yield batch_sample_features, batch_mask, batch_origin_entity_pair_ids, \
                  batch_origin_entity_pair_ids_in_path, np.array(batch_sample_label), batch_entity_type_ids

    def path_aggregator(self, e_r_embedding, input, mask):
        # mask: batch_size x max_path_mun x max_path_len
        # e_r_embedding: batch x max_path_num x max_path_len x embedding_size

        max_path_len = e_r_embedding.shape[2]
        relation_idx = list(range(1, max_path_len, 2))
        r_embedding = tf.reduce_sum(tf.gather(e_r_embedding, relation_idx, axis=2), axis=-2)# batch x max_path_num x embedding_size
        path_existing = tf.expand_dims(tf.reduce_sum(mask, axis=-1)>0, axis=-1) # batch_size x max_path_mun x 1
        # geometric_relation_emb = r_embedding * tf.cast(path_existing, dtype=tf.float32)# batch x max_path_num x embedding_size

        # print(tf.reduce_sum(mask, axis=-1)>0)

        # h = self.agg_fc(e_r_embedding) + input  # batch x max_path_num x max_path_len x feature_size + input
        # # h = tf.reduce_sum(r_embedding, axis=2) # batch x max_path_num x relation_num x feature_size
        # h = tf.reduce_sum(h, axis=2) # batch x max_path_num x embedding_size
        path_weight = self.path_weight_mat1(r_embedding) # batch x max_path_num x 1
        r_embedding = self.agg_fc1(r_embedding)  # batch x max_path_num/2 x feature_size
        path_weight = tf.nn.softmax(tf.where(path_existing, path_weight, tf.ones_like(path_weight) * (-1e10)), axis=1)
        # print(path_weight)
        path_agg_relation_emb = tf.reduce_sum(path_weight * r_embedding, axis=1) # batch x feature_size
        return path_agg_relation_emb, r_embedding, path_weight

    def entity_type_encoder(self, batch_entity_type_ids):

        # print(batch_entity_type_ids)
        # batch_size x max_path_mun x max_entity_num x max_type_num_per_entity x embedding_size
        select_entity_type_embedding = tf.gather(self.entity_type_embedding_table, batch_entity_type_ids)
        # print(select_entity_type_embedding)

        entity_type_weight = self.entityType_weight_mat(select_entity_type_embedding)# batch_size x max_path_mun x max_entity_num x max_type_num_per_entity x 1
        # print(entity_type_weight)
        # print(tf.expand_dims(batch_entity_type_ids, axis=-1))

        entity_type_weight = tf.where(tf.expand_dims(batch_entity_type_ids, axis=-1)==self.entity_type_num, -1e10*tf.ones_like(entity_type_weight), entity_type_weight)
        # print(entity_type_weight)
        entity_type_weight = tf.nn.softmax(entity_type_weight, axis=-2)

        h = select_entity_type_embedding * entity_type_weight # batch_size x max_path_mun x max_entity_num x max_type_num_per_entity x embedding_size
        h = tf.reduce_sum(h, axis=-2)# batch_size x max_path_mun x max_entity_num x embedding_size

        # batch_size x max_path_mun x max_entity_num x max_type_num_per_entity*64
        # select_entity_type_embedding = tf.squeeze(tf.concat(tf.split(select_entity_type_embedding, select_entity_type_embedding.shape[-2], axis=-2), axis=-1), axis=-2)
        # h = self.entityType_fc(select_entity_type_embedding)# batch_size x max_path_mun x max_entity_num x embedding_size
        # print(h)
        flatten_h = tf.reshape(h, (h.shape[0], h.shape[1], 1, -1)) # batch_size x max_path_mun x 1 x max_entity_num*embedding_size
        flatten_h = tf.concat([flatten_h, tf.zeros_like(flatten_h)], axis=-2)# batch_size x max_path_mun x 2 x max_entity_num*embedding_size
        h = tf.concat(tf.split(flatten_h, h.shape[-2], axis=-1), axis=-2)[:,:,:-1,:]
        # zeros = tf.zeros((h.shape[0], h.shape[1], h.shape[2]-1, h.shape[3]))
        # h = tf.concat([h, zeros], axis=-2)# batch_size x max_path_mun x max_entity_num x 2*embedding_size
        # print(flatten_h)

        return h

    def geometric_relation(self, e_r_embedding, batch_origin_entity_pair_ids_in_path, mask):
        # e_r_embedding: batch x max_path_num x max_path_len x embedding_size
        # batch_origin_entity_pair_ids_in_path: batch x max_path_num x max_path_len
        # mask: batch_size x max_path_mun x max_path_len
        batch_origin_entity_pair_ids_in_path = tf.cast(batch_origin_entity_pair_ids_in_path, dtype=tf.float32)
        batch_origin_entity_pair_ids_in_path = tf.expand_dims(batch_origin_entity_pair_ids_in_path, axis=-1) # batch x max_path_num x max_path_len x 1
        head_tail_embedding = e_r_embedding * batch_origin_entity_pair_ids_in_path #batch x max_path_num x max_path_len x embedding_size
        geometric_relation_emb = tf.reduce_sum(head_tail_embedding, axis=-2) #batch x max_path_num x embedding_size
        path_existing = tf.expand_dims(tf.reduce_sum(mask, axis=-1) > 0, axis=-1)  # batch_size x max_path_mun x 1

        path_weight = self.path_weight_mat2(geometric_relation_emb)  # batch x max_path_num x 1
        r_embedding = self.agg_fc2(geometric_relation_emb)  # batch x max_path_num/2 x feature_size
        path_weight = tf.nn.softmax(tf.where(path_existing, path_weight, tf.ones_like(path_weight) * (-1e10)), axis=1)
        # print(path_weight)
        geometric_relation_emb = tf.reduce_sum(path_weight * r_embedding, axis=1)  # batch x feature_size
        return geometric_relation_emb, r_embedding, path_weight

    def call(self, input, mask, batch_origin_entity_pair_ids, batch_entity_type_ids, batch_origin_entity_pair_ids_in_path=None, training=None):
        # batch_path_entity_flags_set: batch_size x max_path_mun x max_entity_num
        # batch_entity_type_ids: batch_size x max_path_mun x max_entity_num x max_type_num_per_entity
        # print(batch_path_entity_flags_set)
        # print(batch_path_entity_flags_set.shape)

        entityType_embedding = self.entity_type_encoder(batch_entity_type_ids)# batch_size x max_path_mun x max_entity_num x embedding_size
        # input = input + entityType_embedding

        # origin_entity_pair_embeddings = 0
        # origin_entity_pair_embeddings = tf.gather(self.embedding_table, batch_origin_entity_pair_ids)
        # print(origin_entity_pair_embeddings)
        h = self.attention_layer1(input, entityType_embedding, mask, training) # batch x max_path_num x max_path_len x embedding_size
        # h = self.attention_layer2(h, mask, entityType_embedding, training) # batch x max_path_num x max_path_len x embedding_size
        e_r_embedding = self.attention_layer3(h, entityType_embedding, mask, training) # batch x max_path_num x max_path_len x embedding_size

        path_agg_relation_emb, path_r_embedding, path_weight1 = self.path_aggregator(e_r_embedding, input, mask)

        geo_agg_relation_emb, geo_r_embedding, path_weight2 = self.geometric_relation(e_r_embedding, batch_origin_entity_pair_ids_in_path, mask) #batch x embedding_size

        # geometric_relation_emb = self.agg_fc_2(geometric_relation_emb)
        if training == True:
            return path_agg_relation_emb, geo_agg_relation_emb, path_r_embedding, geo_r_embedding
        else:
            return path_agg_relation_emb, geo_agg_relation_emb, [path_weight1, path_weight2, path_r_embedding, geo_r_embedding]

    def compute_macro_MIG_loss(self, agg_relation_emb, geometric_relation_emb, y_true):
        # y_true = tf.reshape(y_true, (1, -1))[0]
        # geometric_relation_emb1 = tf.abs(geometric_relation_emb1)# batch x max_path_num x embedding_size
        # geometric_relation_emb2 = tf.abs(geometric_relation_emb2)# batch x max_path_num x embedding_size
        batch_num = geometric_relation_emb.shape[0]
        if batch_num > 1:
            I = tf.cast(np.eye(batch_num), dtype=tf.float32)
            E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
            normalize_1 = batch_num
            normalize_2 = batch_num * (batch_num - 1)

            # new_output = geometric_relation_emb
            m = tf.matmul(geometric_relation_emb, agg_relation_emb, transpose_b=True)# batch x max_path_num x max_path_num
            # print(m)
            noise = np.random.rand(1) * 0.0001
            # print(m * I + I * noise + E - I)
            m1 = tf.math.log(m * I + I * noise + E - I) # + I * noise  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
            # m1 = tf.math.log(m * I + E - I)
            # print(m1)
            m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
            loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        else:
            I = tf.cast(np.eye(batch_num), dtype=tf.float32)
            E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
            normalize_1 = batch_num

            # new_output = geometric_relation_emb / tf.math.sqrt(self.feature_size)
            m = tf.matmul(geometric_relation_emb, agg_relation_emb, transpose_b=True)
            # print(m)
            noise = np.random.rand(1) * 0.0001
            # print(m * I + I * noise + E - I)
            m1 = tf.math.log(m * I + I * noise + E - I)  # + I * noise  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息

            loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1

        return loss #pos_score - 0.001*neg_score

    def compute_micro_MIG_loss(self, agg_relation_emb, geometric_relation_emb, y_true):
        # y_true = tf.reshape(y_true, (1, -1))[0]
        # print(path_r_embedding.shape, geo_r_embedding.shape)
        # geometric_relation_emb1 = tf.abs(geometric_relation_emb1)# batch x max_path_num x embedding_size
        # geometric_relation_emb2 = tf.abs(geometric_relation_emb2)# batch x max_path_num x embedding_size
        batch = geometric_relation_emb.shape[0]
        max_path_num = geometric_relation_emb.shape[1]

        E = tf.cast(np.ones((batch, max_path_num, max_path_num)), dtype=tf.float32)
        I = E * tf.expand_dims(tf.cast(np.eye(max_path_num), dtype=tf.float32), axis=0)
        normalize_1 = max_path_num
        normalize_2 = max_path_num * (max_path_num - 1)

        # new_output = geometric_relation_emb
        m = tf.matmul(geometric_relation_emb, agg_relation_emb, transpose_b=True)  # batch x max_path_num x max_path_num
        ones = tf.ones_like(m)
        m = tf.where(m == 0, ones, m)
        # print(m)
        noise = np.random.rand(1) * 0.0001
        # print(m * I + I * noise + E - I)
        m1 = tf.math.log(m * I + I * noise + E - I)  # + I * noise  # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)
        # print(m1)
        if max_path_num > 1:
            m2 = m * (E - I)  # i<->j，最小化P(i,j)互信息
            loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + max_path_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
        else:
            loss = -(tf.reduce_sum(tf.reduce_sum(m1)) + max_path_num) / normalize_1

        return loss/batch #pos_score - 0.001*neg_score

    def compute_clf_loss(self, agg_relation_emb, geometric_relation_emb, relation_embedding_table, y_true, task, vars):

        # y_true = tf.one_hot(y_true, relation_num)
        # print(one_hot)
        if task == 'all':
            score = tf.nn.softmax(tf.matmul(agg_relation_emb, relation_embedding_table, transpose_b=True), axis=-1)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y_true, axis=-1)
        else:

            # head_entity_embeddings = tf.squeeze(tf.gather(origin_entity_pair_embeddings, [0], axis=1), axis=1)
            # tail_entity_embeddings = tf.squeeze(tf.gather(origin_entity_pair_embeddings, [1], axis=1), axis=1)
            #
            # q1 = self.agg_fc_2(tail_entity_embeddings - head_entity_embeddings)

            score1 = tf.matmul(agg_relation_emb, relation_embedding_table, transpose_b=True)
            score2 = tf.matmul(geometric_relation_emb, relation_embedding_table, transpose_b=True)

            loss = 0.5*tf.nn.sigmoid_cross_entropy_with_logits(logits=score1, labels=y_true) + 0.5*tf.nn.sigmoid_cross_entropy_with_logits(logits=score2, labels=y_true)

            # loss = 0.0 * tf.nn.sigmoid_cross_entropy_with_logits(logits=score1, labels=y_true) + tf.nn.sigmoid_cross_entropy_with_logits(logits=score2, labels=y_true)

        kernel_vals = [var for var in vars if "kernel" in var.name]
        l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
        return tf.reduce_sum(loss) + tf.add_n(l2_losses) * 5e-4

def calculate_hit(agg_relation_emb, target, relation_embedding_table, batch_sample_label, topK):
    score = tf.matmul(agg_relation_emb, relation_embedding_table, transpose_b=True) # batch x 400
    score_ranking = tf.argsort(score, direction='DESCENDING', axis=-1)
    topK_score = tf.gather(score_ranking, topK, axis=-1)# batch x k
    mask = tf.ones_like(topK_score) * np.reshape(target, (-1, 1))# batch x k
    # print(mask)
    hit_all = tf.cast(tf.equal(mask, topK_score), tf.int32)# batch x k
    hit_all = tf.reduce_sum(hit_all, axis=-1)# batch x 1
    pos_mask = tf.cast(batch_sample_label, dtype=tf.bool)
    hit = tf.boolean_mask(hit_all, pos_mask, axis=0)
    hit_sum = tf.reduce_sum(hit)
    # hit = tf.reduce_sum(tf.cast(tf.equal(hit_all, tf.reshape(batch_sample_label, (-1, 1))), tf.int32))

    return hit_sum

def calculate_AP(y_true, score):
    batch_AP = average_precision_score(y_true, score)
    return batch_AP

def mean_average_precision(agg_path_emb, selected_relation_embedding_table, all_labels):
    '''
    Args:
        all_probs: 2D numpy.ndarray. The first dimension is number of samples, and the second one is the number of class
        all_labels: 1D numpy.ndarray. Save the index of labels.
    '''

    all_probs = tf.nn.softmax(tf.matmul(agg_path_emb, tf.cast(selected_relation_embedding_table, tf.float32), transpose_b=True), axis=-1)
    def make_one_hot(input_array, num_class):
        out_array = np.eye(num_class)[input_array]
        return out_array

    n_sample, num_class = all_probs.shape

    all_labels = make_one_hot(all_labels, num_class)
    n_precision = []
    for each_class in range(num_class):
        probs = all_probs[:, each_class]
        labels = all_labels[:, each_class]
        order = np.argsort(-probs)  # Sort by confidence from largest to smallest
        # probs = probs[order]
        labels = labels[order]
        precision = []
        recall = []
        for i in range(n_sample):
            pos_pred_label = labels[0:i + 1]
            neg_pred_label = labels[i + 1:]
            tp = np.sum(pos_pred_label)
            fp = len(pos_pred_label) - tp
            fn = np.sum(neg_pred_label)
            P = tp / (tp + fp + 1e-10)
            R = tp / (tp + fn + 1e-10)
            precision.append(P)
            recall.append(R)
        recall_change_index_0 = []  # The same recall value may correspond multiple precision values. So we take the largest precision value.
        for i in range(n_sample - 1):
            if recall[i] != recall[i + 1]:
                recall_change_index_0.append(i + 1)
        recall_change_index_1 = recall_change_index_0[0:]
        recall_change_index_0.insert(0, 0)
        recall_change_index_1.append(n_sample)
        precision = np.array(precision)
        recall = np.array(recall)
        for i in range(len(recall_change_index_1)):
            index_0 = recall_change_index_0[i]
            index_1 = recall_change_index_1[i]
            precision[index_0:index_1] = np.max(precision[index_0:])
        unique_precision = []
        unique_precision.append(precision[0])
        for i in range(n_sample - 1):
            if recall[i] != recall[i + 1]:  # Only take precision when recall changes
                unique_precision.append(precision[i + 1])
        n_precision.append(np.mean(unique_precision))
    return n_precision

    # mAP = np.mean(np.array(n_precision))
    # return mAP

def topK(k):
    return list(range(k))

'''
concept_agentbelongstoorganization
concept_athletehomestadium
concept_athleteplaysforteam
concept_athleteplaysinleague
concept_athleteplayssport
concept_organizationheadquarteredincity
concept_organizationhiredperson
concept_personborninlocation
concept_personleadsorganization,
concept_teamplaysinleague
concept_teamplayssport
concept_worksfor
'''

'''
    base@schemastaging@organization_extra@phone_number.@base@schemastaging@phone_sandbox@service_location 48.58
	education@educational_institution@school_type 42.85
	film@film@language
	film@film@music 51.25
	film@film@written_by 52.58
	location@location@contains 46.29
	medicine@symptom@symptom_of 92.38
	music@artist@origin 48.85
	organization@organization_founder@organizations_founded 59.81
	people@ethnicity@languages_spoken 44.36
	people@person@nationality 80.81
	people@person@place_of_birth 44.72
	people@profession@specialization_of 72.39+
	sports@sports_team@roster.@american_football@football_roster_position@position 40.03+
	sports@sports_team@sport 84.66
	sports@sports_team_location@teams 90.48+
	time@event@locations 61.48+
	tv@tv_program@country_of_origin 88.70+
	tv@tv_program@genre 44.42
	tv@tv_program@languages 92.70+
'''

dataset_name = sys.argv[1] #'FB15k-237' 'NELL-995'
task = sys.argv[2] #'concept_organizationheadquarteredincity'
max_path_mun = sys.argv[3] # 10
split_path_num = sys.argv[4]
batch_size = 32
dataset = DataLoader(dataset_name=dataset_name, task=task, batch_size=batch_size, shuffle=True, max_path_mun=int(max_path_mun), split_path_num=int(split_path_num))
entity_num = dataset.entity_num
relation_num = dataset.relation_num
relation_ids = dataset.relation_ids
# relation_embedding_table = dataset.relation_embedding_table
relationIDs, taskIDs = dataset.relationID2taskID() #(relationIDs, taskIDs)
# print(dataset.entity_type_num)
# print(relationIDs, taskIDs)

model = Model(dataset=dataset,
              feature_size=dataset.embedding_table.shape[-1],
              embedding_size=256, attention_head=8, drop_rate=0.2)

initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                             decay_steps=dataset.batch_num_train,
                                                             decay_rate=0.98)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# topK = list(range(1))
epochs = 5
best_hit_1 = 0
best_hit_3 = 0
best_hit_5 = 0
best_hit_10 = 0
best_mAP = 0
best_f1 = 0
for epoch in range(epochs):
    train_dataIterator = model.train_loader()
    test_dataIterator = model.test_loader()
    total_loss = 0
    print('Task: ' + task)
    for step in range(dataset.batch_num_train):
        with tf.GradientTape() as tape:
            batch_sample_features, batch_mask, batch_origin_entity_pair_ids, batch_origin_entity_pair_ids_in_path, batch_sample_label, batch_entity_type_ids = next(train_dataIterator)
            path_agg_relation_emb, geo_agg_relation_emb, path_r_embedding, geo_r_embedding = model(batch_sample_features, batch_mask,
                                                                batch_origin_entity_pair_ids,
                                                                batch_entity_type_ids,
                                                                batch_origin_entity_pair_ids_in_path,
                                                                training=True)
            # selected_task_ids = [taskIDs[relationIDs.index(relation_id)] for relation_id in batch_sample_label - entity_num]
            # y_true = tf.one_hot(selected_task_ids, len(taskIDs))
            # print(agg_path_emb.shape)
            # print(origin_entity_pair_embeddings.shape)
            if task == 'all':
                y_true = tf.one_hot(batch_sample_label - entity_num, relation_num)
                loss = model.compute_clf_loss(path_agg_relation_emb, model.embedding_table[model.entity_num:], y_true, task)
            else:
                task_id = relationIDs[0] #relation_ids[':'.join(task.split('_'))] - entity_num
                # print(task_id)
                y_true = tf.cast(np.reshape(batch_sample_label, (-1,1)), dtype=tf.float32)
                # print(y_true)
                # origin_entity_pair_embeddings = tf.gather(model.embedding_table, batch_origin_entity_pair_ids)
                loss1 = model.compute_clf_loss(path_agg_relation_emb,
                                               geo_agg_relation_emb,
                                               tf.gather(model.embedding_table[entity_num:], [int(task_id)]),
                                               y_true, task, tape.watched_variables())

                loss_glob = model.compute_macro_MIG_loss(path_agg_relation_emb, geo_agg_relation_emb, y_true)
                loss_loc = model.compute_micro_MIG_loss(path_r_embedding, geo_r_embedding, y_true)

                # print(loss1)
                # print(loss2)
                # print('.................')
                loss = loss1 + loss_glob + loss_loc
            # print(y_true)

            total_loss += loss

            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
        if step % 20 == 0:
            print("epoch = {}\tstep = {}\tlearning rate = {}\tloss = {}".format(epoch, step, optimizer._decayed_lr(tf.float32), loss))
    print("total_loss = {}".format(total_loss))

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    all_y_true = []
    all_prb = []
    test_pairs = defaultdict(lambda : defaultdict(int))
    all_predict = {}
    all_pos_num = 0

    bi_score = []
    bi_gt = []
    pred_rel = []
    for step in range(dataset.batch_num_test):
        batch_sample_features, batch_mask, batch_origin_entity_pair_ids, batch_origin_entity_pair_ids_in_path, batch_sample_label, batch_entity_type_ids = next(test_dataIterator)
        path_agg_relation_emb, geo_agg_relation_emb, parameters = model(batch_sample_features, batch_mask,
                                                                        batch_origin_entity_pair_ids,
                                                                        batch_entity_type_ids,
                                                                        batch_origin_entity_pair_ids_in_path,
                                                                        training=False)

        path_weight1, path_weight2, path_r_embedding, geo_r_embedding = parameters
        score = tf.matmul(path_agg_relation_emb + geo_agg_relation_emb, model.embedding_table[entity_num:], transpose_b=True)  # batch x 400
        score_ranking = tf.argsort(score, direction='DESCENDING', axis=-1)
        topK_score = tf.gather(score_ranking, topK(1), axis=-1)
        pred_rel.append(topK_score)
        #
        print(path_weight1)
        print(path_weight2)
        # print(relationIDs[0], topK_score)
        # print(path_r_embedding)
        # print(geo_r_embedding)

        pred_relation_emb = path_agg_relation_emb + geo_agg_relation_emb
        if task == 'all':
            targets = batch_sample_label - entity_num

            hit_1 += calculate_hit(pred_relation_emb, targets, model.embedding_table[entity_num:], batch_sample_label, topK(1))

            selected_task_ids = [taskIDs[relationIDs.index(relation_id)] for relation_id in targets]
            # APs += mean_average_precision(agg_path_emb, tf.gather(relation_embedding_table, relationIDs), selected_task_ids)

            selected_relation_embedding_table = tf.gather(model.embedding_table[entity_num:], relationIDs)
            all_y_true.append(tf.one_hot(selected_task_ids, len(taskIDs)))
            all_prb.append(tf.nn.softmax(tf.matmul(pred_relation_emb, tf.cast(selected_relation_embedding_table, tf.float32), transpose_b=True), axis=-1))
        else:
            task_id = relationIDs[0]
            hit_1 += calculate_hit(pred_relation_emb, task_id, model.embedding_table[entity_num:], batch_sample_label, topK(1))
            hit_3 += calculate_hit(pred_relation_emb, task_id, model.embedding_table[entity_num:], batch_sample_label, topK(3))
            hit_5 += calculate_hit(pred_relation_emb, task_id, model.embedding_table[entity_num:], batch_sample_label, topK(5))
            hit_10 += calculate_hit(pred_relation_emb, task_id, model.embedding_table[entity_num:], batch_sample_label, topK(10))
            all_pos_num += np.sum(batch_sample_label)
            # print(batch_origin_entity_pair_ids.shape)

            batch_predict = tf.nn.sigmoid(tf.matmul(pred_relation_emb, tf.gather(model.embedding_table[entity_num:], [int(task_id)], tf.float32), transpose_b=True))
            for i in range(len(batch_origin_entity_pair_ids)):
                e1, e2 = batch_origin_entity_pair_ids[i]
                all_predict[(e1, e2)] = batch_predict[i][0]
                test_pairs[e1][e2] = batch_sample_label[i]
                if batch_sample_label[i] == 1:
                    # pred = np.argmax([1-batch_predict[i][0], batch_predict[i][0]])
                    # bi_score.append(pred)
                    bi_score.append(batch_predict[i][0])
                    bi_gt.append(batch_sample_label[i])
                else:
                    # pred = np.argmax([batch_predict[i][0], 1 - batch_predict[i][0]])
                    # bi_score.append(pred)
                    bi_score.append(batch_predict[i][0])
                    bi_gt.append(batch_sample_label[i])

    # idx = np.reshape(np.where(np.array(bi_gt)==1)[0], (-1,1))
    # bool_mask = tf.cast(bi_gt, dtype=tf.bool)
    # mask_gt = tf.reshape(tf.boolean_mask(np.array(bi_gt) * task_id, bool_mask), (-1,1))
    # pred_rel = tf.boolean_mask(tf.concat(pred_rel, axis=0), bool_mask)
    # trip = np.concatenate([idx, np.array(mask_gt), np.array(pred_rel)], axis=-1)
    # np.savetxt('case.csv', trip)
    # print(trip)

    fpr, tpr, thresholds = roc_curve(bi_gt, bi_score, pos_label=1)
    f1 = auc(fpr, tpr)
    # print("-----sklearn:", auc(fpr, tpr))

    hit_rate_1 = hit_1 / all_pos_num
    hit_rate_3 = hit_3 / all_pos_num
    hit_rate_5 = hit_5 / all_pos_num
    hit_rate_10 = hit_10 / all_pos_num
    # mAP = calculate_AP(tf.concat(all_y_true, axis=0), tf.concat(all_prb, axis=0))

    aps = []
    score_all = []
    # calculate MAP
    for e1 in test_pairs:
        y_true = []
        y_score = []
        for e2 in test_pairs[e1]:
            score = all_predict[(e1, e2)]
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)

        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mAP = np.mean(aps)

    if hit_rate_1 > best_hit_1:
        best_hit_1 = hit_rate_1
    if hit_rate_3 > best_hit_3:
        best_hit_3 = hit_rate_3
    if hit_rate_5 > best_hit_5:
        best_hit_5 = hit_rate_5
    if hit_rate_10 > best_hit_10:
        best_hit_10 = hit_rate_10
    if f1 > best_f1:
        best_f1 = f1
    if mAP > best_mAP:
        best_mAP = mAP

    print("epoch = {}\tmAP = {}\tbest mAP = {}".format(epoch, mAP, best_mAP))
    print("epoch = {}\tf1 = {}\tbest f1 = {}".format(epoch, f1, best_f1))
    # print("epoch = {}\thit_rate_1 = {}\tbest_hit_1 = {}".format(epoch, hit_rate_1, best_hit_1))
    # print("epoch = {}\thit_rate_3 = {}\tbest_hit_3 = {}".format(epoch, hit_rate_3, best_hit_3))
    # print("epoch = {}\thit_rate_5 = {}\tbest_hit_5 = {}".format(epoch, hit_rate_5, best_hit_5))
    # print("epoch = {}\thit_rate_10 = {}\tbest_hit_10 = {}".format(epoch, hit_rate_10, best_hit_10))
    print('='*90)

# homestadium
# [[  21   55  212]
#  [  34   55  133]
#  [  78   55   55]
#  [ 119   55   55]
#  [ 136   55   55]
#  [ 145   55   55]
#  [ 147   55   55]
#  [ 157   55   55]
#  [ 160   55   55]
#  [ 165   55   55]
#  [ 166   55   55]
#  [ 176   55   55]
#  [ 181   55   55]
#  [ 183   55  133]
#  [ 187   55   55]
#  [ 191   55   55]
#  [ 201   55   55]
#  [ 202   55   55]
#  [ 208   55   55]
#  [ 216   55   55]
#  [ 226   55   55]
#  [ 243   55   55]
#  [ 248   55  133]
#  [ 257   55   55]
#  [ 291   55   55]
#  [ 298   55   55]
#  [ 307   55   55]
#  [ 324   55   55]
#  [ 346   55   55]
#  [ 347   55   55] +
#  [ 352   55   55]
#  [ 363   55   55]
#  [ 374   55   55] +
#  [ 393   55   55]
#  [ 399   55   55]
#  [ 405   55   55]
#  [ 406   55  146]
#  [ 417   55  133]
#  [ 428   55   55]
#  [ 437   55  342]
#  [ 445   55   55]
#  [ 469   55   55]
#  [ 477   55   55]
#  [ 488   55   55]
#  [ 506   55   55]
#  [ 515   55   55] +
#  [ 517   55   55]
#  [ 520   55   55]
#  [ 542   55   55]
#  [ 548   55   55]
#  [ 550   55   55]
#  [ 566   55   55]
#  [ 578   55   55]
#  [ 580   55  133]
#  [ 584   55   55]
#  [ 586   55   55]
#  [ 593   55   55]
#  [ 595   55   55]
#  [ 603   55   55]
#  [ 619   55   55]
#  [ 625   55  133]
#  [ 628   55   55]
#  [ 629   55  133]
#  [ 662   55   55]
#  [ 675   55   55]
#  [ 680   55   55]
#  [ 683   55   55]
#  [ 689   55   55]
#  [ 705   55   55]
#  [ 707   55   55]
#  [ 712   55   55]
#  [ 722   55   55]
#  [ 727   55   55]
#  [ 757   55  133]
#  [ 769   55   55]
#  [ 813   55  133]
#  [ 824   55  133]
#  [ 827   55   55]
#  [ 839   55   55]
#  [ 858   55   55]
#  [ 865   55   55]
#  [ 872   55   55]
#  [ 882   55   55]
#  [ 893   55  199]
#  [ 900   55   55]
#  [ 908   55  212]
#  [ 927   55  212]
#  [ 928   55   55]
#  [ 950   55  133]
#  [ 969   55   55]
#  [ 972   55   55]
#  [ 983   55   55]
#  [ 990   55  199]
#  [1000   55   55]
#  [1001   55   55]
#  [1008   55   55]
#  [1013   55   55]
#  [1017   55   55]
#  [1018   55   55]
#  [1023   55  133]
#  [1028   55   55]
#  [1032   55   55]
#  [1054   55   55]
#  [1057   55   55]
#  [1062   55   55]
#  [1067   55   55]
#  [1079   55   55]
#  [1085   55   55]
#  [1088   55  133]
#  [1089   55   55]
#  [1125   55   55]
#  [1129   55   55]
#  [1134   55   55]
#  [1150   55   55]
#  [1154   55   55]
#  [1155   55   55]
#  [1160   55   55]
#  [1163   55   55]
#  [1207   55   55]
#  [1218   55   55]
#  [1232   55   55]
#  [1233   55   55]
#  [1240   55   55]
#  [1250   55   55]
#  [1251   55  133]
#  [1265   55  199]
#  [1275   55   55]
#  [1279   55   55]
#  [1302   55   55]
#  [1303   55   55]
#  [1305   55   55]
#  [1309   55   55]
#  [1318   55   55]
#  [1320   55  102]
#  [1333   55   55]
#  [1334   55   55]
#  [1361   55   55]
#  [1364   55   55]
#  [1368   55   55]
#  [1392   55   55]
#  [1395   55   55]
#  [1401   55   55]
#  [1420   55   55]
#  [1429   55   55]
#  [1437   55   55]
#  [1442   55   55]
#  [1443   55   55]
#  [1454   55   55]
#  [1487   55   55]
#  [1506   55   55]
#  [1517   55   55]
#  [1532   55   55]
#  [1534   55   55]
#  [1546   55   55]
#  [1557   55   55]
#  [1558   55   55]
#  [1561   55   55]
#  [1571   55   55]
#  [1573   55   55]
#  [1583   55   55]
#  [1620   55   55]
#  [1632   55   55]
#  [1638   55   55]
#  [1641   55   55]
#  [1650   55   55]
#  [1652   55   55]
#  [1654   55   55]
#  [1660   55   55]
#  [1673   55   55]
#  [1679   55   55]
#  [1681   55   55]
#  [1693   55   55]
#  [1707   55   55]
#  [1721   55  133]
#  [1724   55   55]
#  [1740   55   55]
#  [1745   55   55]
#  [1752   55   55]
#  [1761   55  146]
#  [1767   55   55]
#  [1771   55   55]
#  [1776   55   55]
#  [1787   55   55]
#  [1799   55   55]
#  [1807   55   55]
#  [1818   55   55]
#  [1820   55   55]
#  [1823   55   55]
#  [1831   55   55]
#  [1832   55   55]
#  [1834   55   55]
#  [1840   55   55]
#  [1861   55   55]
#  [1865   55   55]
#  [1884   55   55]
#  [1887   55   55]
#  [1893   55   55]
#  [1906   55   55]
#  [1910   55   55]
#  [1912   55   55]
#  [1928   55   55]]

'''
    "0": {
        "origin entity pair": [
            "concept_athlete_ryan_giggs",
            "concept_stadiumoreventvenue_old_trafford"
        ],
        "sample_paths": [
            [
                [
                    "concept_athlete_ryan_giggs",
                    "concept_sportsteam_man_utd",
                    "concept_stadiumoreventvenue_old_trafford"
                ],
                [
                    "concept:athleteplaysforteam",
                    "concept:teamhomestadium"
                ]
            ],
            [
                [
                    "concept_athlete_ryan_giggs",
                    "concept_sportsteam_man_utd",
                    "concept_dateliteral_n2006",
                    "concept_country_norway",
                    "concept_company_national",
                    "concept_city_manchester",
                    "concept_stadiumoreventvenue_old_trafford"
                ],
                [
                    "concept:athleteplaysforteam",
                    "concept:atdate",
                    "concept:atdate_inv",
                    "concept:hasofficeincountry_inv",
                    "concept:hasofficeincity",
                    "concept:stadiumlocatedincity_inv"
                ]
            ],
            [
                [
                    "concept_athlete_ryan_giggs",
                    "concept_sportsteam_man_utd",
                    "concept_dateliteral_n2006",
                    "concept_country_united_states",
                    "concept_stateorprovince_iowa",
                    "concept_city_cedar_rapids",
                    "concept_company_national",
                    "concept_city_manchester",
                    "concept_stadiumoreventvenue_old_trafford"
                ],
                [
                    "concept:athleteplaysforteam",
                    "concept:atdate",
                    "concept:atdate_inv",
                    "concept:statelocatedingeopoliticallocation_inv",
                    "concept:locationlocatedwithinlocation_inv",
                    "concept:hasofficeincity_inv",
                    "concept:hasofficeincity",
                    "concept:stadiumlocatedincity_inv"
                ]
            ]
        ],
        "label": 1
    }

tf.Tensor(
[[[9.9999964e-01]
  [3.1904679e-07]
  [2.6857865e-08]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]]], shape=(1, 10, 1), dtype=float32)
tf.Tensor(
[[[0.5711375 ]
  [0.21349381]
  [0.2153687 ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]]], shape=(1, 10, 1), dtype=float32)
'''

'''
    "0": {
        "origin entity pair": [
            "concept_athlete_clay_buchholz",
            "concept_stadiumoreventvenue_fenway"
        ],
        "sample_paths": [
            [
                [
                    "concept_athlete_clay_buchholz",
                    "concept_sportsteam_red_sox",
                    "concept_stadiumoreventvenue_fenway"
                ],
                [
                    "concept:athleteplaysforteam",
                    "concept:teamhomestadium"
                ]
            ],
            [
                [
                    "concept_athlete_clay_buchholz",
                    "concept_sportsteamposition_center",
                    "concept_personmexico_julio_franco",
                    "concept_stadiumoreventvenue_fenway"
                ],
                [
                    "concept:athleteflyouttosportsteamposition",
                    "concept:athleteflyouttosportsteamposition_inv",
                    "concept:athletehomestadium"
                ]
            ],
            [
                [
                    "concept_athlete_clay_buchholz",
                    "concept_sport_baseball",
                    "concept_country_usa",
                    "concept_stateorprovince_kentucky",
                    "concept_city_owensboro",
                    "concept_coach_kentucky",
                    "concept_sportsteam_ncaa_youth_kids",
                    "concept_sportsteam_louisville_cardinals",
                    "concept_sportsteam_red_sox",
                    "concept_stadiumoreventvenue_fenway"
                ],
                [
                    "concept:athleteplayssport",
                    "concept:sportfansincountry",
                    "concept:statelocatedincountry_inv",
                    "concept:locationlocatedwithinlocation_inv",
                    "concept:mutualproxyfor_inv",
                    "concept:agentcompeteswithagent_inv",
                    "concept:subpartoforganization_inv",
                    "concept:teamplaysagainstteam_inv",
                    "concept:teamhomestadium"
                ]
            ]
        ],
        "label": 1
    }
    
tf.Tensor(
[[[9.6780908e-01]
  [3.2190982e-02]
  [1.5154287e-09]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]]], shape=(1, 10, 1), dtype=float32)
tf.Tensor(
[[[0.873244  ]
  [0.08453628]
  [0.04221975]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]]], shape=(1, 10, 1), dtype=float32)

'''

'''
    "0": {
        "origin entity pair": [
            "concept_athlete_jay_payton",
            "concept_stadiumoreventvenue_camden_yards"
        ],
        "sample_paths": [
            [
                [
                    "concept_athlete_jay_payton",
                    "concept_sportsteam_orioles",
                    "concept_sportsteam_new_york_mets",
                    "concept_county_los_angeles_county",
                    "concept_company_nbc",
                    "concept_city_washington_d_c",
                    "concept_stadiumoreventvenue_rfk_memorial_stadium",
                    "concept_sport_baseball",
                    "concept_stadiumoreventvenue_camden_yards"
                ],
                [
                    "concept:athleteplaysforteam",
                    "concept:teamplaysagainstteam_inv",
                    "concept:teamplaysincity",
                    "concept:hasofficeincity_inv",
                    "concept:hasofficeincity",
                    "concept:stadiumlocatedincity_inv",
                    "concept:sportusesstadium_inv",
                    "concept:sportusesstadium"
                ]
            ],
            [
                [
                    "concept_athlete_jay_payton",
                    "concept_sport_baseball",
                    "concept_stadiumoreventvenue_camden_yards"
                ],
                [
                    "concept:athleteplayssport",
                    "concept:sportusesstadium"
                ]
            ],
            [
                [
                    "concept_athlete_jay_payton",
                    "concept_sportsteamposition_center",
                    "concept_sport_football",
                    "concept_awardtrophytournament_super_bowl",
                    "concept_sportsteam_bengals",
                    "concept_sportsleague_mlb",
                    "concept_stadiumoreventvenue_camden_yards"
                ],
                [
                    "concept:athleteflyouttosportsteamposition",
                    "concept:sporthassportsteamposition_inv",
                    "concept:awardtrophytournamentisthechampionshipgameofthenationalsport_inv",
                    "concept:teamwontrophy_inv",
                    "concept:teamplaysinleague",
                    "concept:leaguestadiums"
                ]
            ],
            [
                [
                    "concept_athlete_jay_payton",
                    "concept_sportsteamposition_center",
                    "concept_coach_peter_moylan",
                    "concept_nerve_hand",
                    "concept_athlete_aaron",
                    "concept_sport_baseball",
                    "concept_stadiumoreventvenue_camden_yards"
                ],
                [
                    "concept:athleteflyouttosportsteamposition",
                    "concept:athleteflyouttosportsteamposition_inv",
                    "concept:athleteinjuredhisbodypart",
                    "concept:athleteinjuredhisbodypart_inv",
                    "concept:athleteplayssport",
                    "concept:sportusesstadium"
                ]
            ]
        ],
        "label": 1
    }

tf.Tensor(
[[[2.8783822e-06]
  [9.9998236e-01]
  [4.9990954e-06]
  [9.7786660e-06]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]
  [0.0000000e+00]]], shape=(1, 10, 1), dtype=float32)
tf.Tensor(
[[[0.35387096]
  [0.22526541]
  [0.21000317]
  [0.21086048]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]
  [0.        ]]], shape=(1, 10, 1), dtype=float32)
'''

