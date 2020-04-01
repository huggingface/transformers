import sentencepiece as spm
p2 = '/Users/shleifer/sentencepiece/src/m2.model'
import sentencepiece_model_pb2 as model # protobuf magic
import copy

def load_model(pth):
    m = model.ModelProto()
    m.ParseFromString(open(pth, 'rb').read())
    return m

def save_model(m, pth):
    with open(pth, 'wb') as f:
        f.write(m.SerializeToString())


m = load_model('m.model') # m = model.ModelProto()
bos_piece, unk_piece, eos_piece = m.pieces[:3]
for piece in bos_piece, unk_piece, eos_piece: m.pieces.remove(piece)
print(m.pieces[:5])
pad_piece = copy.deepcopy(bos_piece)
pad_piece.piece = '<pad>'
new_order = reversed([bos_piece, pad_piece, eos_piece, unk_piece])
for new_piece in new_order: m.pieces.insert(0, new_piece)
print(m.pieces[:5])
# Add the language codes:

save_model(m, 'm2.model')


LANGS = 'ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN'
lang_codes = LANGS.split(',')
for code in lang_codes:
    new_piece = copy.deepcopy(bos_piece) # piece.type=control
    new_piece.piece = code
    m.pieces.append(new_piece)
