import json
from konlpy.tag import Okt

okt = Okt()

josa = ['은', '는', '에서', '가', '의', '으로', '을', '에', '이', '에게', '에는', '라는', '를', '에', '와', '과', '아', '야', '께', '께서']

with open('./predictions.json') as pred:
   out_dic = dict()
   pred = json.load(pred)

   for k, v in pred.items():
       _ = v.split()
       r = okt.pos(_.pop(-1))
       last = r[-1]
       if last[1] == 'Josa':
           if last[0] in josa:
               r.pop(-1)
               print(last)
       add = ''.join([x for x, y in r])
       _.append(add)
       new = ' '.join(_)
       out_dic[k] = new

   print(len(pred))
   print(len(out_dic))
   with open('./predictions2.json', 'w') as new:
       json.dump(out_dic, new, ensure_ascii=False, indent=2)
