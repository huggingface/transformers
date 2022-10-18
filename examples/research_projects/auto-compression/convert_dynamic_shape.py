import argparse
import paddle
import math
from x2paddle.op_mapper.pytorch2paddle import pytorch_custom_layer as x2paddle_nn
from datasets import load_dataset

class BertForSequenceClassification(paddle.nn.Layer):
    def __init__(self, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert_embeddings_position_ids = self.create_parameter(dtype='int64', shape=(1, 512), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.embedding0 = paddle.nn.Embedding(num_embeddings=28996, embedding_dim=768, padding_idx=0)
        self.embedding1 = paddle.nn.Embedding(num_embeddings=2, embedding_dim=768)
        self.embedding2 = paddle.nn.Embedding(num_embeddings=512, embedding_dim=768)
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear0 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear2 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax0 = paddle.nn.Softmax()
        self.dropout1 = paddle.nn.Dropout(p=0.0)
        self.linear3 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout2 = paddle.nn.Dropout(p=0.0)
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear4 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu0 = paddle.nn.GELU()
        self.linear5 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout3 = paddle.nn.Dropout(p=0.0)
        self.layernorm2 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear6 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear7 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear8 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax1 = paddle.nn.Softmax()
        self.dropout4 = paddle.nn.Dropout(p=0.0)
        self.linear9 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout5 = paddle.nn.Dropout(p=0.0)
        self.layernorm3 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear10 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu1 = paddle.nn.GELU()
        self.linear11 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout6 = paddle.nn.Dropout(p=0.0)
        self.layernorm4 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear12 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear13 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear14 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax2 = paddle.nn.Softmax()
        self.dropout7 = paddle.nn.Dropout(p=0.0)
        self.linear15 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout8 = paddle.nn.Dropout(p=0.0)
        self.layernorm5 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear16 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu2 = paddle.nn.GELU()
        self.linear17 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout9 = paddle.nn.Dropout(p=0.0)
        self.layernorm6 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear18 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear19 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear20 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax3 = paddle.nn.Softmax()
        self.dropout10 = paddle.nn.Dropout(p=0.0)
        self.linear21 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout11 = paddle.nn.Dropout(p=0.0)
        self.layernorm7 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear22 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu3 = paddle.nn.GELU()
        self.linear23 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout12 = paddle.nn.Dropout(p=0.0)
        self.layernorm8 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear24 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear25 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear26 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax4 = paddle.nn.Softmax()
        self.dropout13 = paddle.nn.Dropout(p=0.0)
        self.linear27 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout14 = paddle.nn.Dropout(p=0.0)
        self.layernorm9 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear28 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu4 = paddle.nn.GELU()
        self.linear29 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout15 = paddle.nn.Dropout(p=0.0)
        self.layernorm10 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear30 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear31 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear32 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax5 = paddle.nn.Softmax()
        self.dropout16 = paddle.nn.Dropout(p=0.0)
        self.linear33 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout17 = paddle.nn.Dropout(p=0.0)
        self.layernorm11 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear34 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu5 = paddle.nn.GELU()
        self.linear35 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout18 = paddle.nn.Dropout(p=0.0)
        self.layernorm12 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear36 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear37 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear38 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax6 = paddle.nn.Softmax()
        self.dropout19 = paddle.nn.Dropout(p=0.0)
        self.linear39 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout20 = paddle.nn.Dropout(p=0.0)
        self.layernorm13 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear40 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu6 = paddle.nn.GELU()
        self.linear41 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout21 = paddle.nn.Dropout(p=0.0)
        self.layernorm14 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear42 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear43 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear44 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax7 = paddle.nn.Softmax()
        self.dropout22 = paddle.nn.Dropout(p=0.0)
        self.linear45 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout23 = paddle.nn.Dropout(p=0.0)
        self.layernorm15 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear46 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu7 = paddle.nn.GELU()
        self.linear47 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout24 = paddle.nn.Dropout(p=0.0)
        self.layernorm16 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear48 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear49 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear50 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax8 = paddle.nn.Softmax()
        self.dropout25 = paddle.nn.Dropout(p=0.0)
        self.linear51 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout26 = paddle.nn.Dropout(p=0.0)
        self.layernorm17 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear52 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu8 = paddle.nn.GELU()
        self.linear53 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout27 = paddle.nn.Dropout(p=0.0)
        self.layernorm18 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear54 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear55 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear56 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax9 = paddle.nn.Softmax()
        self.dropout28 = paddle.nn.Dropout(p=0.0)
        self.linear57 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout29 = paddle.nn.Dropout(p=0.0)
        self.layernorm19 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear58 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu9 = paddle.nn.GELU()
        self.linear59 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout30 = paddle.nn.Dropout(p=0.0)
        self.layernorm20 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear60 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear61 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear62 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax10 = paddle.nn.Softmax()
        self.dropout31 = paddle.nn.Dropout(p=0.0)
        self.linear63 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout32 = paddle.nn.Dropout(p=0.0)
        self.layernorm21 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear64 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu10 = paddle.nn.GELU()
        self.linear65 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout33 = paddle.nn.Dropout(p=0.0)
        self.layernorm22 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear66 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear67 = paddle.nn.Linear(in_features=768, out_features=768)
        self.linear68 = paddle.nn.Linear(in_features=768, out_features=768)
        self.softmax11 = paddle.nn.Softmax()
        self.dropout34 = paddle.nn.Dropout(p=0.0)
        self.linear69 = paddle.nn.Linear(in_features=768, out_features=768)
        self.dropout35 = paddle.nn.Dropout(p=0.0)
        self.layernorm23 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear70 = paddle.nn.Linear(in_features=768, out_features=3072)
        self.gelu11 = paddle.nn.GELU()
        self.linear71 = paddle.nn.Linear(in_features=3072, out_features=768)
        self.dropout36 = paddle.nn.Dropout(p=0.0)
        self.layernorm24 = paddle.nn.LayerNorm(normalized_shape=[768], epsilon=1e-12)
        self.linear72 = paddle.nn.Linear(in_features=768, out_features=768)
        self.tanh0 = paddle.nn.Tanh()
        self.dropout37 = paddle.nn.Dropout(p=0.0)
        self.linear73 = paddle.nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, x0, x1, x2):
        x5 = 8.0
        x6 = -2
        x7 = -1
        x11 = -3.4028234663852886e+38
        x12 = 1.0
        x18 = 1
        x20 = 0
        x20_list = [0]
        x19_list = [2147483647]
        x18_list = [1]
        x24 = paddle.strided_slice(x=x1, axes=x20_list, starts=x20_list, ends=x19_list, strides=x18_list)
        x25 = paddle.unsqueeze(x=x24, axis=1)
        x26 = paddle.unsqueeze(x=x25, axis=2)
        x16_list = [3]
        x20_list = [0]
        x19_list = [2147483647]
        x18_list = [1]
        x27 = paddle.strided_slice(x=x26, axes=x16_list, starts=x20_list, ends=x19_list, strides=x18_list)
        x28 = paddle.cast(x=x27, dtype='float32')
        x29 = x12 - x28 * x18
        x30 = x29 * x11
        bert_embeddings_position_ids = self.bert_embeddings_position_ids
        x20_list = [0]
        x19_list = [2147483647]
        x18_list = [1]
        x37 = paddle.strided_slice(x=bert_embeddings_position_ids, axes=x20_list, starts=x20_list, ends=x19_list, strides=x18_list)
        x20_list = [0]
        x36_list = [128]
        x18_list = [1]
        x38 = paddle.strided_slice(x=x37, axes=x18_list, starts=x20_list, ends=x36_list, strides=x18_list)
        x40 = self.embedding0(x0)
        x42 = self.embedding1(x2)
        x43 = x40 + x42
        x45 = self.embedding2(x38)
        x46 = x43 + x45
        x50 = self.layernorm0(x46)
        x51 = self.dropout0(x50)
        x88 = self.linear0(x51)
        x93 = self.linear1(x51)
        x95 = paddle.reshape(x=x93, shape=[-1, 128, 12, 64])
        x97 = paddle.transpose(x=x95, perm=[0, 2, 1, 3])
        x102 = self.linear2(x51)
        x104 = paddle.reshape(x=x102, shape=[-1, 128, 12, 64])
        x106 = paddle.transpose(x=x104, perm=[0, 2, 1, 3])
        x108 = paddle.reshape(x=x88, shape=[-1, 128, 12, 64])
        x110 = paddle.transpose(x=x108, perm=[0, 2, 1, 3])
        x111_shape = x97.shape
        x111_len = len(x111_shape)
        x111_list = []
        for i in range(x111_len):
            x111_list.append(i)
        if x7 < 0:
            x7_new = x7 + x111_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x111_len
        else:
            x6_new = x6
        x111_list[x7_new] = x6_new
        x111_list[x6_new] = x7_new
        x111 = paddle.transpose(x=x97, perm=x111_list)
        x112 = paddle.matmul(x=x110, y=x111)
        x113 = x112 / x5
        x114 = x113 + x30
        x115 = self.softmax0(x114)
        x116 = self.dropout1(x115)
        x117 = paddle.matmul(x=x116, y=x106)
        x119 = paddle.transpose(x=x117, perm=[0, 2, 1, 3])
        x120 = x119
        x122 = paddle.reshape(x=x120, shape=[-1, 128, 768])
        x129 = self.linear3(x122)
        x130 = self.dropout2(x129)
        x131 = x130 + x51
        x135 = self.layernorm1(x131)
        x141 = self.linear4(x135)
        x142 = self.gelu0(x141)
        x149 = self.linear5(x142)
        x150 = self.dropout3(x149)
        x151 = x150 + x135
        x155 = self.layernorm2(x151)
        x168 = self.linear6(x155)
        x173 = self.linear7(x155)
        x175 = paddle.reshape(x=x173, shape=[-1, 128, 12, 64])
        x177 = paddle.transpose(x=x175, perm=[0, 2, 1, 3])
        x182 = self.linear8(x155)
        x184 = paddle.reshape(x=x182, shape=[-1, 128, 12, 64])
        x186 = paddle.transpose(x=x184, perm=[0, 2, 1, 3])
        x188 = paddle.reshape(x=x168, shape=[-1, 128, 12, 64])
        x190 = paddle.transpose(x=x188, perm=[0, 2, 1, 3])
        x191_shape = x177.shape
        x191_len = len(x191_shape)
        x191_list = []
        for i in range(x191_len):
            x191_list.append(i)
        if x7 < 0:
            x7_new = x7 + x191_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x191_len
        else:
            x6_new = x6
        x191_list[x7_new] = x6_new
        x191_list[x6_new] = x7_new
        x191 = paddle.transpose(x=x177, perm=x191_list)
        x192 = paddle.matmul(x=x190, y=x191)
        x193 = x192 / x5
        x194 = x193 + x30
        x195 = self.softmax1(x194)
        x196 = self.dropout4(x195)
        x197 = paddle.matmul(x=x196, y=x186)
        x199 = paddle.transpose(x=x197, perm=[0, 2, 1, 3])
        x200 = x199
        x202 = paddle.reshape(x=x200, shape=[-1, 128, 768])
        x209 = self.linear9(x202)
        x210 = self.dropout5(x209)
        x211 = x210 + x155
        x215 = self.layernorm3(x211)
        x221 = self.linear10(x215)
        x222 = self.gelu1(x221)
        x229 = self.linear11(x222)
        x230 = self.dropout6(x229)
        x231 = x230 + x215
        x235 = self.layernorm4(x231)
        x248 = self.linear12(x235)
        x253 = self.linear13(x235)
        x255 = paddle.reshape(x=x253, shape=[-1, 128, 12, 64])
        x257 = paddle.transpose(x=x255, perm=[0, 2, 1, 3])
        x262 = self.linear14(x235)
        x264 = paddle.reshape(x=x262, shape=[-1, 128, 12, 64])
        x266 = paddle.transpose(x=x264, perm=[0, 2, 1, 3])
        x268 = paddle.reshape(x=x248, shape=[-1, 128, 12, 64])
        x270 = paddle.transpose(x=x268, perm=[0, 2, 1, 3])
        x271_shape = x257.shape
        x271_len = len(x271_shape)
        x271_list = []
        for i in range(x271_len):
            x271_list.append(i)
        if x7 < 0:
            x7_new = x7 + x271_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x271_len
        else:
            x6_new = x6
        x271_list[x7_new] = x6_new
        x271_list[x6_new] = x7_new
        x271 = paddle.transpose(x=x257, perm=x271_list)
        x272 = paddle.matmul(x=x270, y=x271)
        x273 = x272 / x5
        x274 = x273 + x30
        x275 = self.softmax2(x274)
        x276 = self.dropout7(x275)
        x277 = paddle.matmul(x=x276, y=x266)
        x279 = paddle.transpose(x=x277, perm=[0, 2, 1, 3])
        x280 = x279
        x282 = paddle.reshape(x=x280, shape=[-1, 128, 768])
        x289 = self.linear15(x282)
        x290 = self.dropout8(x289)
        x291 = x290 + x235
        x295 = self.layernorm5(x291)
        x301 = self.linear16(x295)
        x302 = self.gelu2(x301)
        x309 = self.linear17(x302)
        x310 = self.dropout9(x309)
        x311 = x310 + x295
        x315 = self.layernorm6(x311)
        x328 = self.linear18(x315)
        x333 = self.linear19(x315)
        x335 = paddle.reshape(x=x333, shape=[-1, 128, 12, 64])
        x337 = paddle.transpose(x=x335, perm=[0, 2, 1, 3])
        x342 = self.linear20(x315)
        x344 = paddle.reshape(x=x342, shape=[-1, 128, 12, 64])
        x346 = paddle.transpose(x=x344, perm=[0, 2, 1, 3])
        x348 = paddle.reshape(x=x328, shape=[-1, 128, 12, 64])
        x350 = paddle.transpose(x=x348, perm=[0, 2, 1, 3])
        x351_shape = x337.shape
        x351_len = len(x351_shape)
        x351_list = []
        for i in range(x351_len):
            x351_list.append(i)
        if x7 < 0:
            x7_new = x7 + x351_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x351_len
        else:
            x6_new = x6
        x351_list[x7_new] = x6_new
        x351_list[x6_new] = x7_new
        x351 = paddle.transpose(x=x337, perm=x351_list)
        x352 = paddle.matmul(x=x350, y=x351)
        x353 = x352 / x5
        x354 = x353 + x30
        x355 = self.softmax3(x354)
        x356 = self.dropout10(x355)
        x357 = paddle.matmul(x=x356, y=x346)
        x359 = paddle.transpose(x=x357, perm=[0, 2, 1, 3])
        x360 = x359
        x362 = paddle.reshape(x=x360, shape=[-1, 128, 768])
        x369 = self.linear21(x362)
        x370 = self.dropout11(x369)
        x371 = x370 + x315
        x375 = self.layernorm7(x371)
        x381 = self.linear22(x375)
        x382 = self.gelu3(x381)
        x389 = self.linear23(x382)
        x390 = self.dropout12(x389)
        x391 = x390 + x375
        x395 = self.layernorm8(x391)
        x408 = self.linear24(x395)
        x413 = self.linear25(x395)
        x415 = paddle.reshape(x=x413, shape=[-1, 128, 12, 64])
        x417 = paddle.transpose(x=x415, perm=[0, 2, 1, 3])
        x422 = self.linear26(x395)
        x424 = paddle.reshape(x=x422, shape=[-1, 128, 12, 64])
        x426 = paddle.transpose(x=x424, perm=[0, 2, 1, 3])
        x428 = paddle.reshape(x=x408, shape=[-1, 128, 12, 64])
        x430 = paddle.transpose(x=x428, perm=[0, 2, 1, 3])
        x431_shape = x417.shape
        x431_len = len(x431_shape)
        x431_list = []
        for i in range(x431_len):
            x431_list.append(i)
        if x7 < 0:
            x7_new = x7 + x431_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x431_len
        else:
            x6_new = x6
        x431_list[x7_new] = x6_new
        x431_list[x6_new] = x7_new
        x431 = paddle.transpose(x=x417, perm=x431_list)
        x432 = paddle.matmul(x=x430, y=x431)
        x433 = x432 / x5
        x434 = x433 + x30
        x435 = self.softmax4(x434)
        x436 = self.dropout13(x435)
        x437 = paddle.matmul(x=x436, y=x426)
        x439 = paddle.transpose(x=x437, perm=[0, 2, 1, 3])
        x440 = x439
        x442 = paddle.reshape(x=x440, shape=[-1, 128, 768])
        x449 = self.linear27(x442)
        x450 = self.dropout14(x449)
        x451 = x450 + x395
        x455 = self.layernorm9(x451)
        x461 = self.linear28(x455)
        x462 = self.gelu4(x461)
        x469 = self.linear29(x462)
        x470 = self.dropout15(x469)
        x471 = x470 + x455
        x475 = self.layernorm10(x471)
        x488 = self.linear30(x475)
        x493 = self.linear31(x475)
        x495 = paddle.reshape(x=x493, shape=[-1, 128, 12, 64])
        x497 = paddle.transpose(x=x495, perm=[0, 2, 1, 3])
        x502 = self.linear32(x475)
        x504 = paddle.reshape(x=x502, shape=[-1, 128, 12, 64])
        x506 = paddle.transpose(x=x504, perm=[0, 2, 1, 3])
        x508 = paddle.reshape(x=x488, shape=[-1, 128, 12, 64])
        x510 = paddle.transpose(x=x508, perm=[0, 2, 1, 3])
        x511_shape = x497.shape
        x511_len = len(x511_shape)
        x511_list = []
        for i in range(x511_len):
            x511_list.append(i)
        if x7 < 0:
            x7_new = x7 + x511_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x511_len
        else:
            x6_new = x6
        x511_list[x7_new] = x6_new
        x511_list[x6_new] = x7_new
        x511 = paddle.transpose(x=x497, perm=x511_list)
        x512 = paddle.matmul(x=x510, y=x511)
        x513 = x512 / x5
        x514 = x513 + x30
        x515 = self.softmax5(x514)
        x516 = self.dropout16(x515)
        x517 = paddle.matmul(x=x516, y=x506)
        x519 = paddle.transpose(x=x517, perm=[0, 2, 1, 3])
        x520 = x519
        x522 = paddle.reshape(x=x520, shape=[-1, 128, 768])
        x529 = self.linear33(x522)
        x530 = self.dropout17(x529)
        x531 = x530 + x475
        x535 = self.layernorm11(x531)
        x541 = self.linear34(x535)
        x542 = self.gelu5(x541)
        x549 = self.linear35(x542)
        x550 = self.dropout18(x549)
        x551 = x550 + x535
        x555 = self.layernorm12(x551)
        x568 = self.linear36(x555)
        x573 = self.linear37(x555)
        x575 = paddle.reshape(x=x573, shape=[-1, 128, 12, 64])
        x577 = paddle.transpose(x=x575, perm=[0, 2, 1, 3])
        x582 = self.linear38(x555)
        x584 = paddle.reshape(x=x582, shape=[-1, 128, 12, 64])
        x586 = paddle.transpose(x=x584, perm=[0, 2, 1, 3])
        x588 = paddle.reshape(x=x568, shape=[-1, 128, 12, 64])
        x590 = paddle.transpose(x=x588, perm=[0, 2, 1, 3])
        x591_shape = x577.shape
        x591_len = len(x591_shape)
        x591_list = []
        for i in range(x591_len):
            x591_list.append(i)
        if x7 < 0:
            x7_new = x7 + x591_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x591_len
        else:
            x6_new = x6
        x591_list[x7_new] = x6_new
        x591_list[x6_new] = x7_new
        x591 = paddle.transpose(x=x577, perm=x591_list)
        x592 = paddle.matmul(x=x590, y=x591)
        x593 = x592 / x5
        x594 = x593 + x30
        x595 = self.softmax6(x594)
        x596 = self.dropout19(x595)
        x597 = paddle.matmul(x=x596, y=x586)
        x599 = paddle.transpose(x=x597, perm=[0, 2, 1, 3])
        x600 = x599
        x602 = paddle.reshape(x=x600, shape=[-1, 128, 768])
        x609 = self.linear39(x602)
        x610 = self.dropout20(x609)
        x611 = x610 + x555
        x615 = self.layernorm13(x611)
        x621 = self.linear40(x615)
        x622 = self.gelu6(x621)
        x629 = self.linear41(x622)
        x630 = self.dropout21(x629)
        x631 = x630 + x615
        x635 = self.layernorm14(x631)
        x648 = self.linear42(x635)
        x653 = self.linear43(x635)
        x655 = paddle.reshape(x=x653, shape=[-1, 128, 12, 64])
        x657 = paddle.transpose(x=x655, perm=[0, 2, 1, 3])
        x662 = self.linear44(x635)
        x664 = paddle.reshape(x=x662, shape=[-1, 128, 12, 64])
        x666 = paddle.transpose(x=x664, perm=[0, 2, 1, 3])
        x668 = paddle.reshape(x=x648, shape=[-1, 128, 12, 64])
        x670 = paddle.transpose(x=x668, perm=[0, 2, 1, 3])
        x671_shape = x657.shape
        x671_len = len(x671_shape)
        x671_list = []
        for i in range(x671_len):
            x671_list.append(i)
        if x7 < 0:
            x7_new = x7 + x671_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x671_len
        else:
            x6_new = x6
        x671_list[x7_new] = x6_new
        x671_list[x6_new] = x7_new
        x671 = paddle.transpose(x=x657, perm=x671_list)
        x672 = paddle.matmul(x=x670, y=x671)
        x673 = x672 / x5
        x674 = x673 + x30
        x675 = self.softmax7(x674)
        x676 = self.dropout22(x675)
        x677 = paddle.matmul(x=x676, y=x666)
        x679 = paddle.transpose(x=x677, perm=[0, 2, 1, 3])
        x680 = x679
        x682 = paddle.reshape(x=x680, shape=[-1, 128, 768])
        x689 = self.linear45(x682)
        x690 = self.dropout23(x689)
        x691 = x690 + x635
        x695 = self.layernorm15(x691)
        x701 = self.linear46(x695)
        x702 = self.gelu7(x701)
        x709 = self.linear47(x702)
        x710 = self.dropout24(x709)
        x711 = x710 + x695
        x715 = self.layernorm16(x711)
        x728 = self.linear48(x715)
        x733 = self.linear49(x715)
        x735 = paddle.reshape(x=x733, shape=[-1, 128, 12, 64])
        x737 = paddle.transpose(x=x735, perm=[0, 2, 1, 3])
        x742 = self.linear50(x715)
        x744 = paddle.reshape(x=x742, shape=[-1, 128, 12, 64])
        x746 = paddle.transpose(x=x744, perm=[0, 2, 1, 3])
        x748 = paddle.reshape(x=x728, shape=[-1, 128, 12, 64])
        x750 = paddle.transpose(x=x748, perm=[0, 2, 1, 3])
        x751_shape = x737.shape
        x751_len = len(x751_shape)
        x751_list = []
        for i in range(x751_len):
            x751_list.append(i)
        if x7 < 0:
            x7_new = x7 + x751_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x751_len
        else:
            x6_new = x6
        x751_list[x7_new] = x6_new
        x751_list[x6_new] = x7_new
        x751 = paddle.transpose(x=x737, perm=x751_list)
        x752 = paddle.matmul(x=x750, y=x751)
        x753 = x752 / x5
        x754 = x753 + x30
        x755 = self.softmax8(x754)
        x756 = self.dropout25(x755)
        x757 = paddle.matmul(x=x756, y=x746)
        x759 = paddle.transpose(x=x757, perm=[0, 2, 1, 3])
        x760 = x759
        x762 = paddle.reshape(x=x760, shape=[-1, 128, 768])
        x769 = self.linear51(x762)
        x770 = self.dropout26(x769)
        x771 = x770 + x715
        x775 = self.layernorm17(x771)
        x781 = self.linear52(x775)
        x782 = self.gelu8(x781)
        x789 = self.linear53(x782)
        x790 = self.dropout27(x789)
        x791 = x790 + x775
        x795 = self.layernorm18(x791)
        x808 = self.linear54(x795)
        x813 = self.linear55(x795)
        x815 = paddle.reshape(x=x813, shape=[-1, 128, 12, 64])
        x817 = paddle.transpose(x=x815, perm=[0, 2, 1, 3])
        x822 = self.linear56(x795)
        x824 = paddle.reshape(x=x822, shape=[-1, 128, 12, 64])
        x826 = paddle.transpose(x=x824, perm=[0, 2, 1, 3])
        x828 = paddle.reshape(x=x808, shape=[-1, 128, 12, 64])
        x830 = paddle.transpose(x=x828, perm=[0, 2, 1, 3])
        x831_shape = x817.shape
        x831_len = len(x831_shape)
        x831_list = []
        for i in range(x831_len):
            x831_list.append(i)
        if x7 < 0:
            x7_new = x7 + x831_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x831_len
        else:
            x6_new = x6
        x831_list[x7_new] = x6_new
        x831_list[x6_new] = x7_new
        x831 = paddle.transpose(x=x817, perm=x831_list)
        x832 = paddle.matmul(x=x830, y=x831)
        x833 = x832 / x5
        x834 = x833 + x30
        x835 = self.softmax9(x834)
        x836 = self.dropout28(x835)
        x837 = paddle.matmul(x=x836, y=x826)
        x839 = paddle.transpose(x=x837, perm=[0, 2, 1, 3])
        x840 = x839
        x842 = paddle.reshape(x=x840, shape=[-1, 128, 768])
        x849 = self.linear57(x842)
        x850 = self.dropout29(x849)
        x851 = x850 + x795
        x855 = self.layernorm19(x851)
        x861 = self.linear58(x855)
        x862 = self.gelu9(x861)
        x869 = self.linear59(x862)
        x870 = self.dropout30(x869)
        x871 = x870 + x855
        x875 = self.layernorm20(x871)
        x888 = self.linear60(x875)
        x893 = self.linear61(x875)
        x895 = paddle.reshape(x=x893, shape=[-1, 128, 12, 64])
        x897 = paddle.transpose(x=x895, perm=[0, 2, 1, 3])
        x902 = self.linear62(x875)
        x904 = paddle.reshape(x=x902, shape=[-1, 128, 12, 64])
        x906 = paddle.transpose(x=x904, perm=[0, 2, 1, 3])
        x908 = paddle.reshape(x=x888, shape=[-1, 128, 12, 64])
        x910 = paddle.transpose(x=x908, perm=[0, 2, 1, 3])
        x911_shape = x897.shape
        x911_len = len(x911_shape)
        x911_list = []
        for i in range(x911_len):
            x911_list.append(i)
        if x7 < 0:
            x7_new = x7 + x911_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x911_len
        else:
            x6_new = x6
        x911_list[x7_new] = x6_new
        x911_list[x6_new] = x7_new
        x911 = paddle.transpose(x=x897, perm=x911_list)
        x912 = paddle.matmul(x=x910, y=x911)
        x913 = x912 / x5
        x914 = x913 + x30
        x915 = self.softmax10(x914)
        x916 = self.dropout31(x915)
        x917 = paddle.matmul(x=x916, y=x906)
        x919 = paddle.transpose(x=x917, perm=[0, 2, 1, 3])
        x920 = x919
        x922 = paddle.reshape(x=x920, shape=[-1, 128, 768])
        x929 = self.linear63(x922)
        x930 = self.dropout32(x929)
        x931 = x930 + x875
        x935 = self.layernorm21(x931)
        x941 = self.linear64(x935)
        x942 = self.gelu10(x941)
        x949 = self.linear65(x942)
        x950 = self.dropout33(x949)
        x951 = x950 + x935
        x955 = self.layernorm22(x951)
        x968 = self.linear66(x955)
        x973 = self.linear67(x955)
        x975 = paddle.reshape(x=x973, shape=[-1, 128, 12, 64])
        x977 = paddle.transpose(x=x975, perm=[0, 2, 1, 3])
        x982 = self.linear68(x955)
        x984 = paddle.reshape(x=x982, shape=[-1, 128, 12, 64])
        x986 = paddle.transpose(x=x984, perm=[0, 2, 1, 3])
        x988 = paddle.reshape(x=x968, shape=[-1, 128, 12, 64])
        x990 = paddle.transpose(x=x988, perm=[0, 2, 1, 3])
        x991_shape = x977.shape
        x991_len = len(x991_shape)
        x991_list = []
        for i in range(x991_len):
            x991_list.append(i)
        if x7 < 0:
            x7_new = x7 + x991_len
        else:
            x7_new = x7
        if x6 < 0:
            x6_new = x6 + x991_len
        else:
            x6_new = x6
        x991_list[x7_new] = x6_new
        x991_list[x6_new] = x7_new
        x991 = paddle.transpose(x=x977, perm=x991_list)
        x992 = paddle.matmul(x=x990, y=x991)
        x993 = x992 / x5
        x994 = x993 + x30
        x995 = self.softmax11(x994)
        x996 = self.dropout34(x995)
        x997 = paddle.matmul(x=x996, y=x986)
        x999 = paddle.transpose(x=x997, perm=[0, 2, 1, 3])
        x1000 = x999
        x1002 = paddle.reshape(x=x1000, shape=[-1, 128, 768])
        x1009 = self.linear69(x1002)
        x1010 = self.dropout35(x1009)
        x1011 = x1010 + x955
        x1015 = self.layernorm23(x1011)
        x1021 = self.linear70(x1015)
        x1022 = self.gelu11(x1021)
        x1029 = self.linear71(x1022)
        x1030 = self.dropout36(x1029)
        x1031 = x1030 + x1015
        x1035 = self.layernorm24(x1031)
        x20_list = [0]
        x19_list = [2147483647]
        x18_list = [1]
        x1037 = paddle.strided_slice(x=x1035, axes=x20_list, starts=x20_list, ends=x19_list, strides=x18_list)
        x1038 = x1037[:, x20]
        x1042 = self.linear72(x1038)
        x1043 = self.tanh0(x1042)
        x1046 = self.dropout37(x1043)
        x1051 = self.linear73(x1046)
        x1053 = dict()
        x1053['logits'] = x1051
        return x1053

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    args = parser.parse_args()

    raw_datasets = load_dataset("glue", args.task_name)

    if args.task_name != "stsb":
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    params = paddle.load(r'model.pdparams')
    model = BertForSequenceClassification(num_labels)
    model.set_dict(params, use_structured_name=True)
    model.eval()
    ## convert to jit
    sepc_list = list()
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[-1, 128], name="x0", dtype="int64"))
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[-1, 128], name="x1", dtype="int64"))
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[-1, 128], name="x2", dtype="int64"))
    static_model = paddle.jit.to_static(model, input_spec=sepc_list)
    paddle.jit.save(static_model, "model")

