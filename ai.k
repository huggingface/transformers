============================= test session starts ==============================
platform linux -- Python 3.7.9, pytest-6.1.1, py-1.9.0, pluggy-0.13.1
rootdir: /workspace/chanwookim/transformers, configfile: setup.cfg
plugins: hydra-core-1.0.6, flaky-3.7.0
collected 0 items / 1 error

==================================== ERRORS ====================================
________________________ ERROR collecting tests/test.py ________________________
tests/test.py:43: in <module>
    assert False
E   assert False
------------------------------- Captured stdout --------------------------------
force_tokens tensor([ 1263, 10963], device='cuda:0')
force_tokens_2 tensor([7165], device='cuda:0')
model_inputs {'input_ids': tensor([[  464,  5156,   318, 13774,   780]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1]], device='cuda:0')}


INPUT

The baby is crying because
The baby is crying because
The baby is crying because
The baby is crying because
The baby is crying because




>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 1263]
new_seqs []
advance_seq [464, 5156, 318, 13774, 780, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 1263]]
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
PREZIP tensor([[-1.4358e+00,  0.0000e+00,  6.7300e+02],
        [-2.0147e+00,  0.0000e+00,  3.3900e+02],
        [-2.3535e+00,  0.0000e+00,  3.4000e+02],
        [-2.4315e+00,  0.0000e+00,  2.8600e+02],
        [-2.7982e+00,  0.0000e+00,  6.0700e+02],
        [-1.0072e+01,  1.0000e+00,  1.2630e+03],
        [-1.2223e+01,  1.0000e+00,  7.1650e+03]], device='cuda:0')
POSTZIP tensor([[-1.0072e+01,  1.0000e+00,  1.2630e+03],
        [-1.4358e+00,  0.0000e+00,  6.7300e+02],
        [-1.2223e+01,  1.0000e+00,  7.1650e+03],
        [-2.0147e+00,  0.0000e+00,  3.3900e+02],
        [-2.3535e+00,  0.0000e+00,  3.4000e+02],
        [-2.4315e+00,  0.0000e+00,  2.8600e+02],
        [-2.7982e+00,  0.0000e+00,  6.0700e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([-10.0721,  -1.4358, -12.2228,  -2.0147,  -2.3535], device='cuda:0')
>>>>>sent_beam_tokens tensor([1263,  673, 7165,  339,  340], device='cuda:0')
>>>>>sent_beam_indices tensor([0, 0, 0, 0, 0], device='cuda:0')
>>>>constraints_completed [False, False, False, False, False]


OUTPUT

The baby is crying because big
The baby is crying because she
The baby is crying because crazy
The baby is crying because he
The baby is crying because it






INPUT

The baby is crying because big
The baby is crying because she
The baby is crying because crazy
The baby is crying because he
The baby is crying because it




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963]]
advance_seq [464, 5156, 318, 13774, 780, 673, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 7165, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263], [464, 5156, 318, 13774, 780, 673, 7165]]
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 339, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263], [464, 5156, 318, 13774, 780, 673, 7165], [464, 5156, 318, 13774, 780, 7165, 1263]]
advance_seq [464, 5156, 318, 13774, 780, 339, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263], [464, 5156, 318, 13774, 780, 673, 7165], [464, 5156, 318, 13774, 780, 7165, 1263], [464, 5156, 318, 13774, 780, 339, 1263]]
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 340, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263], [464, 5156, 318, 13774, 780, 673, 7165], [464, 5156, 318, 13774, 780, 7165, 1263], [464, 5156, 318, 13774, 780, 339, 1263], [464, 5156, 318, 13774, 780, 339, 7165]]
advance_seq [464, 5156, 318, 13774, 780, 340, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 1263], [464, 5156, 318, 13774, 780, 673, 7165], [464, 5156, 318, 13774, 780, 7165, 1263], [464, 5156, 318, 13774, 780, 339, 1263], [464, 5156, 318, 13774, 780, 339, 7165], [464, 5156, 318, 13774, 780, 340, 1263]]
PREZIP tensor([[-3.3543e+00,  0.0000e+00,  3.3800e+02],
        [-3.7313e+00,  0.0000e+00,  3.3800e+02],
        [-3.7631e+00,  0.0000e+00,  3.7300e+02],
        [-3.8234e+00,  0.0000e+00,  3.3800e+02],
        [-3.8247e+00,  0.0000e+00,  3.1800e+02],
        [-6.6173e+00,  2.0000e+00,  1.0963e+04],
        [-1.2264e+01,  1.0000e+00,  1.2630e+03],
        [-1.3972e+01,  1.0000e+00,  7.1650e+03],
        [-5.7579e+00,  3.0000e+00,  1.2630e+03],
        [-1.1960e+01,  1.0000e+00,  1.2630e+03],
        [-1.3683e+01,  1.0000e+00,  7.1650e+03],
        [-1.2435e+01,  1.0000e+00,  1.2630e+03],
        [-1.3548e+01,  1.0000e+00,  7.1650e+03]], device='cuda:0')
POSTZIP tensor([[-5.7579e+00,  3.0000e+00,  1.2630e+03],
        [-6.6173e+00,  2.0000e+00,  1.0963e+04],
        [-1.1960e+01,  1.0000e+00,  1.2630e+03],
        [-3.3543e+00,  0.0000e+00,  3.3800e+02],
        [-1.2264e+01,  1.0000e+00,  1.2630e+03],
        [-3.7313e+00,  0.0000e+00,  3.3800e+02],
        [-1.2435e+01,  1.0000e+00,  1.2630e+03],
        [-3.7631e+00,  0.0000e+00,  3.7300e+02],
        [-1.3548e+01,  1.0000e+00,  7.1650e+03],
        [-3.8234e+00,  0.0000e+00,  3.3800e+02],
        [-1.3683e+01,  1.0000e+00,  7.1650e+03],
        [-3.8247e+00,  0.0000e+00,  3.1800e+02],
        [-1.3972e+01,  1.0000e+00,  7.1650e+03]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -5.7579,  -6.6173, -11.9598,  -3.3543, -12.2640], device='cuda:0')
>>>>>sent_beam_tokens tensor([ 1263, 10963,  1263,   338,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([2, 0, 3, 1, 1], device='cuda:0')
>>>>constraints_completed [False, False, False, False, False]


OUTPUT

The baby is crying because crazy big
The baby is crying because big monsters
The baby is crying because he big
The baby is crying because she's
The baby is crying because she big






INPUT

The baby is crying because crazy big
The baby is crying because big monsters
The baby is crying because he big
The baby is crying because she's
The baby is crying because she big




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 7165, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 1263, 10963, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 7165, 1263, 10963]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 339, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 7165, 1263, 10963], [464, 5156, 318, 13774, 780, 1263, 10963, 7165]]
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 7165, 1263, 10963], [464, 5156, 318, 13774, 780, 1263, 10963, 7165], [464, 5156, 318, 13774, 780, 339, 1263, 10963]]
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 7165, 1263, 10963], [464, 5156, 318, 13774, 780, 1263, 10963, 7165], [464, 5156, 318, 13774, 780, 339, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 1263]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 7165, 1263, 10963], [464, 5156, 318, 13774, 780, 1263, 10963, 7165], [464, 5156, 318, 13774, 780, 339, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 1263], [464, 5156, 318, 13774, 780, 673, 338, 7165]]
PREZIP tensor([[-5.8691e+00,  0.0000e+00,  5.8700e+02],
        [-6.3662e+00,  0.0000e+00,  4.0700e+02],
        [-6.4227e+00,  0.0000e+00,  1.2008e+04],
        [-6.4356e+00,  0.0000e+00,  7.7870e+03],
        [-6.7641e+00,  0.0000e+00,  5.2300e+02],
        [-5.7414e+00,  4.0000e+00,  1.0963e+04],
        [-1.2468e+01,  4.0000e+00,  7.1650e+03],
        [-1.0331e+01,  2.0000e+00,  1.0963e+04],
        [-8.2352e+00,  1.0000e+00,  1.2630e+03],
        [-7.5897e+00,  1.0000e+00,  7.1650e+03],
        [-1.0610e+01,  2.0000e+00,  1.0963e+04]], device='cuda:0')
POSTZIP tensor([[-5.7414e+00,  4.0000e+00,  1.0963e+04],
        [-1.0331e+01,  2.0000e+00,  1.0963e+04],
        [-7.5897e+00,  1.0000e+00,  7.1650e+03],
        [-5.8691e+00,  0.0000e+00,  5.8700e+02],
        [-1.2468e+01,  4.0000e+00,  7.1650e+03],
        [-1.0610e+01,  2.0000e+00,  1.0963e+04],
        [-8.2352e+00,  1.0000e+00,  1.2630e+03],
        [-6.3662e+00,  0.0000e+00,  4.0700e+02],
        [-6.4227e+00,  0.0000e+00,  1.2008e+04],
        [-6.4356e+00,  0.0000e+00,  7.7870e+03],
        [-6.7641e+00,  0.0000e+00,  5.2300e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -5.7414, -10.3309,  -7.5897,  -5.8691, -12.4677], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963, 10963,  7165,   587,  7165], device='cuda:0')
>>>>>sent_beam_indices tensor([0, 2, 3, 3, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because crazy big monsters
The baby is crying because he big monsters
The baby is crying because she's crazy
The baby is crying because she's been
The baby is crying because big monsters crazy






INPUT

The baby is crying because crazy big monsters
The baby is crying because he big monsters
The baby is crying because she's crazy
The baby is crying because she's been
The baby is crying because big monsters crazy




>>>>>advance_tokens tensor([7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 339, 1263, 10963, 7165]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 339, 1263, 10963, 7165]]
>>>>>advance_tokens tensor([1263, 7165], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 339, 1263, 10963, 7165], [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263]]
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165]
new_seqs [[464, 5156, 318, 13774, 780, 339, 1263, 10963, 7165], [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263], [464, 5156, 318, 13774, 780, 673, 338, 587, 1263]]
PREZIP tensor([[-6.8971e+00,  2.0000e+00,  3.8900e+02],
        [-8.0195e+00,  0.0000e+00,  4.2300e+02],
        [-8.5441e+00,  0.0000e+00,  1.2970e+03],
        [-8.9806e+00,  2.0000e+00,  1.3000e+01],
        [-9.2211e+00,  0.0000e+00,  1.6110e+04],
        [-1.1526e+01,  4.0000e+00,  7.1650e+03],
        [-7.5428e+00,  3.0000e+00,  1.2630e+03],
        [-1.1152e+01,  1.0000e+00,  1.2630e+03],
        [-9.3288e+00,  1.0000e+00,  7.1650e+03]], device='cuda:0')
POSTZIP tensor([[-1.1526e+01,  4.0000e+00,  7.1650e+03],
        [-7.5428e+00,  3.0000e+00,  1.2630e+03],
        [-6.8971e+00,  2.0000e+00,  3.8900e+02],
        [-9.3288e+00,  1.0000e+00,  7.1650e+03],
        [-8.0195e+00,  0.0000e+00,  4.2300e+02],
        [-8.9806e+00,  2.0000e+00,  1.3000e+01],
        [-1.1152e+01,  1.0000e+00,  1.2630e+03],
        [-8.5441e+00,  0.0000e+00,  1.2970e+03],
        [-9.2211e+00,  0.0000e+00,  1.6110e+04]], device='cuda:0')
>>>>>sent_beam_scores tensor([-11.5259,  -7.5428,  -6.8971,  -9.3288,  -8.0195], device='cuda:0')
>>>>>sent_beam_tokens tensor([7165, 1263,  389, 7165,  423], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 2, 0, 3, 0], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because he big monsters crazy
The baby is crying because she's crazy big
The baby is crying because crazy big monsters are
The baby is crying because she's been crazy
The baby is crying because crazy big monsters have






INPUT

The baby is crying because he big monsters crazy
The baby is crying because she's crazy big
The baby is crying because crazy big monsters are
The baby is crying because she's been crazy
The baby is crying because crazy big monsters have




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 10963]]
PREZIP tensor([[-8.8730e+00,  0.0000e+00,  2.4060e+03],
        [-8.9872e+00,  0.0000e+00,  2.9000e+02],
        [-9.2539e+00,  0.0000e+00,  1.3000e+01],
        [-9.5069e+00,  0.0000e+00,  1.1000e+01],
        [-9.6623e+00,  0.0000e+00,  5.5300e+02],
        [-1.2619e+01,  4.0000e+00,  1.0963e+04],
        [-8.8750e+00,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-1.2619e+01,  4.0000e+00,  1.0963e+04],
        [-8.8750e+00,  3.0000e+00,  1.2630e+03],
        [-8.8730e+00,  0.0000e+00,  2.4060e+03],
        [-8.9872e+00,  0.0000e+00,  2.9000e+02],
        [-9.2539e+00,  0.0000e+00,  1.3000e+01],
        [-9.5069e+00,  0.0000e+00,  1.1000e+01],
        [-9.6623e+00,  0.0000e+00,  5.5300e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([-12.6191,  -8.8750,  -8.8730,  -8.9872,  -9.2539], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,  2406,   290,    13], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 3, 2, 1, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, True, False]


OUTPUT

The baby is crying because she's crazy big monsters
The baby is crying because she's been crazy big
The baby is crying because crazy big monsters are coming
The baby is crying because she's crazy big and
The baby is crying because she's crazy big.






INPUT

The baby is crying because she's crazy big monsters
The baby is crying because she's been crazy big
The baby is crying because crazy big monsters are coming
The baby is crying because she's crazy big and
The baby is crying because she's crazy big.




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 13, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 10963]]
PREZIP tensor([[-1.0600e+01,  0.0000e+00,  6.7300e+02],
        [-1.0826e+01,  2.0000e+00,  2.9000e+02],
        [-1.0833e+01,  2.0000e+00,  1.3750e+03],
        [-1.0837e+01,  0.0000e+00,  1.3000e+01],
        [-1.1092e+01,  4.0000e+00,  1.3000e+01],
        [-1.2403e+01,  4.0000e+00,  1.0963e+04],
        [-1.5209e+01,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-1.1092e+01,  4.0000e+00,  1.3000e+01],
        [-1.5209e+01,  3.0000e+00,  1.2630e+03],
        [-1.0826e+01,  2.0000e+00,  2.9000e+02],
        [-1.0600e+01,  0.0000e+00,  6.7300e+02],
        [-1.2403e+01,  4.0000e+00,  1.0963e+04],
        [-1.0833e+01,  2.0000e+00,  1.3750e+03],
        [-1.0837e+01,  0.0000e+00,  1.3000e+01]], device='cuda:0')
>>>>>sent_beam_scores tensor([-11.0919, -15.2090, -10.8264, -10.5999, -12.4034], device='cuda:0')
>>>>>sent_beam_tokens tensor([   13,  1263,   290,   673, 10963], device='cuda:0')
>>>>>sent_beam_indices tensor([2, 4, 1, 3, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because crazy big monsters are coming.
The baby is crying because she's crazy big. big
The baby is crying because she's been crazy big and
The baby is crying because she's crazy big and she
The baby is crying because she's been crazy big monsters






INPUT

The baby is crying because crazy big monsters are coming.
The baby is crying because she's crazy big. big
The baby is crying because she's been crazy big and
The baby is crying because she's crazy big and she
The baby is crying because she's been crazy big monsters




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 13, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 13, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 13, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263]]
PREZIP tensor([[-1.1419e+01,  2.0000e+00,  3.3800e+02],
        [-1.2924e+01,  0.0000e+00,  6.7300e+02],
        [-1.3312e+01,  2.0000e+00,  1.9800e+02],
        [-1.3459e+01,  2.0000e+00,  3.3820e+03],
        [-1.3549e+01,  0.0000e+00,  4.6000e+02],
        [-9.9536e+00,  4.0000e+00,  1.0963e+04],
        [-3.8742e+00,  3.0000e+00,  1.2630e+03],
        [-1.0987e+01,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-9.9536e+00,  4.0000e+00,  1.0963e+04],
        [-3.8742e+00,  3.0000e+00,  1.2630e+03],
        [-1.1419e+01,  2.0000e+00,  3.3800e+02],
        [-1.2924e+01,  0.0000e+00,  6.7300e+02],
        [-1.0987e+01,  3.0000e+00,  1.2630e+03],
        [-1.3312e+01,  2.0000e+00,  1.9800e+02],
        [-1.3549e+01,  0.0000e+00,  4.6000e+02],
        [-1.3459e+01,  2.0000e+00,  3.3820e+03]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -9.9536,  -3.8742, -11.4194, -12.9239, -10.9866], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,   338,   673,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 2, 3, 2, 3], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because she's crazy big. big monsters
The baby is crying because she's been crazy big and big
The baby is crying because she's crazy big and she's
The baby is crying because she's been crazy big and she
The baby is crying because she's crazy big and she big






INPUT

The baby is crying because she's crazy big. big monsters
The baby is crying because she's been crazy big and big
The baby is crying because she's crazy big and she's
The baby is crying because she's been crazy big and she
The baby is crying because she's crazy big and she big




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 673, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 673, 1263]]
PREZIP tensor([[-5.2463e+00,  0.0000e+00,  2.9000e+02],
        [-5.9868e+00,  2.0000e+00,  1.3000e+01],
        [-6.1505e+00,  0.0000e+00,  1.1000e+01],
        [-6.1889e+00,  0.0000e+00,  5.5300e+02],
        [-6.7390e+00,  0.0000e+00,  3.2900e+02],
        [-1.1919e+01,  4.0000e+00,  1.0963e+04],
        [-5.6936e+00,  3.0000e+00,  1.2630e+03],
        [-1.1110e+01,  3.0000e+00,  1.2630e+03],
        [-9.3693e+00,  4.0000e+00,  1.0963e+04]], device='cuda:0')
POSTZIP tensor([[-9.3693e+00,  4.0000e+00,  1.0963e+04],
        [-5.6936e+00,  3.0000e+00,  1.2630e+03],
        [-5.9868e+00,  2.0000e+00,  1.3000e+01],
        [-5.2463e+00,  0.0000e+00,  2.9000e+02],
        [-1.1919e+01,  4.0000e+00,  1.0963e+04],
        [-1.1110e+01,  3.0000e+00,  1.2630e+03],
        [-6.1505e+00,  0.0000e+00,  1.1000e+01],
        [-6.1889e+00,  0.0000e+00,  5.5300e+02],
        [-6.7390e+00,  0.0000e+00,  3.2900e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -9.3693,  -5.6936,  -5.9868,  -5.2463, -11.9188], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,    13,   290, 10963], device='cuda:0')
>>>>>sent_beam_indices tensor([4, 2, 1, 1, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because she's crazy big and she big monsters
The baby is crying because she's crazy big and she's big
The baby is crying because she's been crazy big and big.
The baby is crying because she's been crazy big and big and
The baby is crying because she's been crazy big and big monsters






INPUT

The baby is crying because she's crazy big and she big monsters
The baby is crying because she's crazy big and she's big
The baby is crying because she's been crazy big and big.
The baby is crying because she's been crazy big and big and
The baby is crying because she's been crazy big and big monsters




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 13, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 13, 1263]]
PREZIP tensor([[-6.3694e+00,  0.0000e+00,  1.2630e+03],
        [-6.8644e+00,  2.0000e+00,  2.9000e+02],
        [-7.4970e+00,  2.0000e+00,  1.3750e+03],
        [-7.5834e+00,  2.0000e+00,  6.7300e+02],
        [-7.8789e+00,  0.0000e+00,  1.5760e+03],
        [-1.2507e+01,  4.0000e+00,  1.0963e+04],
        [-1.6012e+01,  3.0000e+00,  1.2630e+03],
        [-1.1232e+00,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-1.2507e+01,  4.0000e+00,  1.0963e+04],
        [-1.1232e+00,  3.0000e+00,  1.2630e+03],
        [-6.8644e+00,  2.0000e+00,  2.9000e+02],
        [-6.3694e+00,  0.0000e+00,  1.2630e+03],
        [-1.6012e+01,  3.0000e+00,  1.2630e+03],
        [-7.4970e+00,  2.0000e+00,  1.3750e+03],
        [-7.8789e+00,  0.0000e+00,  1.5760e+03],
        [-7.5834e+00,  2.0000e+00,  6.7300e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([-12.5067,  -1.1232,  -6.8644,  -6.3694, -16.0120], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,   290,  1263,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 3, 1, 3, 2], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because she's crazy big and she's big monsters
The baby is crying because she's been crazy big and big and big
The baby is crying because she's crazy big and she's big and
The baby is crying because she's been crazy big and big and big
The baby is crying because she's been crazy big and big. big






INPUT

The baby is crying because she's crazy big and she's big monsters
The baby is crying because she's been crazy big and big and big
The baby is crying because she's crazy big and she's big and
The baby is crying because she's been crazy big and big and big
The baby is crying because she's been crazy big and big. big




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 10963]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 13, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 290, 1263]]
PREZIP tensor([[-2.4862e+00,  0.0000e+00,  2.9000e+02],
        [-2.8610e+00,  2.0000e+00,  1.3000e+01],
        [-3.3225e+00,  0.0000e+00,  5.5300e+02],
        [-3.4911e+00,  0.0000e+00,  1.1000e+01],
        [-3.9857e+00,  0.0000e+00,  3.2900e+02],
        [-1.2293e+01,  4.0000e+00,  1.0963e+04],
        [-4.9025e+00,  3.0000e+00,  1.2630e+03],
        [-9.6560e+00,  4.0000e+00,  1.0963e+04]], device='cuda:0')
POSTZIP tensor([[-9.6560e+00,  4.0000e+00,  1.0963e+04],
        [-4.9025e+00,  3.0000e+00,  1.2630e+03],
        [-2.8610e+00,  2.0000e+00,  1.3000e+01],
        [-2.4862e+00,  0.0000e+00,  2.9000e+02],
        [-1.2293e+01,  4.0000e+00,  1.0963e+04],
        [-3.3225e+00,  0.0000e+00,  5.5300e+02],
        [-3.4911e+00,  0.0000e+00,  1.1000e+01],
        [-3.9857e+00,  0.0000e+00,  3.2900e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -9.6560,  -4.9025,  -2.8610,  -2.4862, -12.2932], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,    13,   290, 10963], device='cuda:0')
>>>>>sent_beam_indices tensor([4, 2, 1, 1, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because she's been crazy big and big. big monsters
The baby is crying because she's crazy big and she's big and big
The baby is crying because she's been crazy big and big and big.
The baby is crying because she's been crazy big and big and big and
The baby is crying because she's been crazy big and big and big monsters






INPUT

The baby is crying because she's been crazy big and big. big monsters
The baby is crying because she's crazy big and she's big and big
The baby is crying because she's been crazy big and big and big.
The baby is crying because she's been crazy big and big and big and
The baby is crying because she's been crazy big and big and big monsters




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 290, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 290, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 7165, 1263, 290, 673, 338, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 1263]]
PREZIP tensor([[-3.3472e+00,  0.0000e+00,  1.2630e+03],
        [-4.4136e+00,  0.0000e+00,  1.3750e+03],
        [-5.0307e+00,  2.0000e+00,  8.4300e+02],
        [-5.0600e+00,  0.0000e+00,  3.1400e+02],
        [-5.1705e+00,  0.0000e+00,  1.9800e+02],
        [-1.2517e+01,  4.0000e+00,  1.0963e+04],
        [-1.6495e+01,  3.0000e+00,  1.2630e+03],
        [-8.6100e-01,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-1.2517e+01,  4.0000e+00,  1.0963e+04],
        [-8.6100e-01,  3.0000e+00,  1.2630e+03],
        [-5.0307e+00,  2.0000e+00,  8.4300e+02],
        [-3.3472e+00,  0.0000e+00,  1.2630e+03],
        [-1.6495e+01,  3.0000e+00,  1.2630e+03],
        [-4.4136e+00,  0.0000e+00,  1.3750e+03],
        [-5.0600e+00,  0.0000e+00,  3.1400e+02],
        [-5.1705e+00,  0.0000e+00,  1.9800e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([-12.5170,  -0.8610,  -5.0307,  -3.3472, -16.4952], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,   843,  1263,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 3, 2, 3, 2], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because she's crazy big and she's big and big monsters
The baby is crying because she's been crazy big and big and big and big
The baby is crying because she's been crazy big and big and big. And
The baby is crying because she's been crazy big and big and big and big
The baby is crying because she's been crazy big and big and big. big






INPUT

The baby is crying because she's crazy big and she's big and big monsters
The baby is crying because she's been crazy big and big and big and big
The baby is crying because she's been crazy big and big and big. And
The baby is crying because she's been crazy big and big and big and big
The baby is crying because she's been crazy big and big and big. big




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 843, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 10963]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 843, 1263]]
PREZIP tensor([[-2.1940e+00,  0.0000e+00,  2.9000e+02],
        [-2.4520e+00,  2.0000e+00,  1.3000e+01],
        [-3.2742e+00,  0.0000e+00,  5.5300e+02],
        [-3.3400e+00,  0.0000e+00,  1.1000e+01],
        [-3.6658e+00,  0.0000e+00,  5.2600e+02],
        [-1.2107e+01,  4.0000e+00,  1.0963e+04],
        [-8.4168e+00,  3.0000e+00,  1.2630e+03],
        [-9.5848e+00,  4.0000e+00,  1.0963e+04]], device='cuda:0')
POSTZIP tensor([[-9.5848e+00,  4.0000e+00,  1.0963e+04],
        [-8.4168e+00,  3.0000e+00,  1.2630e+03],
        [-2.4520e+00,  2.0000e+00,  1.3000e+01],
        [-2.1940e+00,  0.0000e+00,  2.9000e+02],
        [-1.2107e+01,  4.0000e+00,  1.0963e+04],
        [-3.2742e+00,  0.0000e+00,  5.5300e+02],
        [-3.3400e+00,  0.0000e+00,  1.1000e+01],
        [-3.6658e+00,  0.0000e+00,  5.2600e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -9.5848,  -8.4168,  -2.4520,  -2.1940, -12.1066], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,    13,   290, 10963], device='cuda:0')
>>>>>sent_beam_indices tensor([4, 2, 1, 1, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because she's been crazy big and big and big. big monsters
The baby is crying because she's been crazy big and big and big. And big
The baby is crying because she's been crazy big and big and big and big.
The baby is crying because she's been crazy big and big and big and big and
The baby is crying because she's been crazy big and big and big and big monsters






INPUT

The baby is crying because she's been crazy big and big and big. big monsters
The baby is crying because she's been crazy big and big and big. And big
The baby is crying because she's been crazy big and big and big and big.
The baby is crying because she's been crazy big and big and big and big and
The baby is crying because she's been crazy big and big and big and big monsters




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 1263]]
PREZIP tensor([[-2.8488e+00,  0.0000e+00,  1.2630e+03],
        [-4.0466e+00,  0.0000e+00,  1.3750e+03],
        [-4.4985e+00,  2.0000e+00,  8.4300e+02],
        [-4.6394e+00,  0.0000e+00,  3.1400e+02],
        [-4.7340e+00,  0.0000e+00,  1.9800e+02],
        [-8.2042e+00,  4.0000e+00,  1.0963e+04],
        [-1.6431e+01,  3.0000e+00,  1.2630e+03],
        [-6.5482e-01,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-8.2042e+00,  4.0000e+00,  1.0963e+04],
        [-6.5482e-01,  3.0000e+00,  1.2630e+03],
        [-4.4985e+00,  2.0000e+00,  8.4300e+02],
        [-2.8488e+00,  0.0000e+00,  1.2630e+03],
        [-1.6431e+01,  3.0000e+00,  1.2630e+03],
        [-4.0466e+00,  0.0000e+00,  1.3750e+03],
        [-4.6394e+00,  0.0000e+00,  3.1400e+02],
        [-4.7340e+00,  0.0000e+00,  1.9800e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -8.2042,  -0.6548,  -4.4985,  -2.8488, -16.4311], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,   843,  1263,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 3, 2, 3, 2], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because she's been crazy big and big and big. And big monsters
The baby is crying because she's been crazy big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big. And
The baby is crying because she's been crazy big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big. big






INPUT

The baby is crying because she's been crazy big and big and big. And big monsters
The baby is crying because she's been crazy big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big. And
The baby is crying because she's been crazy big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big. big




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 843, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 10963]]
>>>>>advance_tokens tensor([10963], device='cuda:0')
>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 1263, 10963]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 843, 1263]]
PREZIP tensor([[-1.7995e+00,  0.0000e+00,  2.9000e+02],
        [-2.2400e+00,  2.0000e+00,  1.3000e+01],
        [-3.2673e+00,  0.0000e+00,  1.1000e+01],
        [-3.4144e+00,  0.0000e+00,  5.5300e+02],
        [-3.4721e+00,  0.0000e+00,  5.2600e+02],
        [-1.1857e+01,  4.0000e+00,  1.0963e+04],
        [-8.4926e+00,  3.0000e+00,  1.2630e+03],
        [-9.7216e+00,  4.0000e+00,  1.0963e+04]], device='cuda:0')
POSTZIP tensor([[-9.7216e+00,  4.0000e+00,  1.0963e+04],
        [-8.4926e+00,  3.0000e+00,  1.2630e+03],
        [-2.2400e+00,  2.0000e+00,  1.3000e+01],
        [-1.7995e+00,  0.0000e+00,  2.9000e+02],
        [-1.1857e+01,  4.0000e+00,  1.0963e+04],
        [-3.2673e+00,  0.0000e+00,  1.1000e+01],
        [-3.4144e+00,  0.0000e+00,  5.5300e+02],
        [-3.4721e+00,  0.0000e+00,  5.2600e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -9.7216,  -8.4926,  -2.2400,  -1.7995, -11.8567], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,    13,   290, 10963], device='cuda:0')
>>>>>sent_beam_indices tensor([4, 2, 1, 1, 1], device='cuda:0')
>>>>constraints_completed [True, False, False, False, True]


OUTPUT

The baby is crying because she's been crazy big and big and big and big. big monsters
The baby is crying because she's been crazy big and big and big and big. And big
The baby is crying because she's been crazy big and big and big and big and big.
The baby is crying because she's been crazy big and big and big and big and big and
The baby is crying because she's been crazy big and big and big and big and big monsters






INPUT

The baby is crying because she's been crazy big and big and big and big. big monsters
The baby is crying because she's been crazy big and big and big and big. And big
The baby is crying because she's been crazy big and big and big and big and big.
The baby is crying because she's been crazy big and big and big and big and big and
The baby is crying because she's been crazy big and big and big and big and big monsters




>>>>>advance_tokens tensor([10963], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963]
new_seqs []
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963]]
>>>>>advance_tokens tensor([1263], device='cuda:0')
advance_seq [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263]
new_seqs [[464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 843, 1263, 10963], [464, 5156, 318, 13774, 780, 673, 338, 587, 7165, 1263, 290, 1263, 290, 1263, 290, 1263, 290, 1263, 13, 1263]]
PREZIP tensor([[-2.2907e+00,  0.0000e+00,  1.2630e+03],
        [-3.8817e+00,  0.0000e+00,  1.3750e+03],
        [-4.2239e+00,  2.0000e+00,  8.4300e+02],
        [-4.4378e+00,  0.0000e+00,  3.1400e+02],
        [-4.4513e+00,  0.0000e+00,  1.9800e+02],
        [-8.3479e+00,  4.0000e+00,  1.0963e+04],
        [-1.6134e+01,  3.0000e+00,  1.2630e+03],
        [-4.9122e-01,  3.0000e+00,  1.2630e+03]], device='cuda:0')
POSTZIP tensor([[-8.3479e+00,  4.0000e+00,  1.0963e+04],
        [-4.9122e-01,  3.0000e+00,  1.2630e+03],
        [-4.2239e+00,  2.0000e+00,  8.4300e+02],
        [-2.2907e+00,  0.0000e+00,  1.2630e+03],
        [-1.6134e+01,  3.0000e+00,  1.2630e+03],
        [-3.8817e+00,  0.0000e+00,  1.3750e+03],
        [-4.4378e+00,  0.0000e+00,  3.1400e+02],
        [-4.4513e+00,  0.0000e+00,  1.9800e+02]], device='cuda:0')
>>>>>sent_beam_scores tensor([ -8.3479,  -0.4912,  -4.2239,  -2.2907, -16.1338], device='cuda:0')
>>>>>sent_beam_tokens tensor([10963,  1263,   843,  1263,  1263], device='cuda:0')
>>>>>sent_beam_indices tensor([1, 3, 2, 3, 2], device='cuda:0')
>>>>constraints_completed [True, False, False, False, False]


OUTPUT

The baby is crying because she's been crazy big and big and big and big. And big monsters
The baby is crying because she's been crazy big and big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big and big. And
The baby is crying because she's been crazy big and big and big and big and big and big
The baby is crying because she's been crazy big and big and big and big and big. big




!! tensor([  464,  5156,   318, 13774,   780,   673,   338,   587,  7165,  1263,
          290,  1263,   290,  1263,   290,  1263,   290,  1263,   290,  1263],
       device='cuda:0')
The baby is crying because she's been crazy big and big and big and big and big and big
!! tensor([  464,  5156,   318, 13774,   780,   673,   338,   587,  7165,  1263,
          290,  1263,   290,  1263,   290,  1263,   290,  1263,   290,  1263],
       device='cuda:0')
The baby is crying because she's been crazy big and big and big and big and big and big
!! tensor([  464,  5156,   318, 13774,   780,   673,   338,   587,  7165,  1263,
          290,  1263,   290,  1263,   290,  1263,   290,  1263,    13,   843],
       device='cuda:0')
The baby is crying because she's been crazy big and big and big and big and big. And
!! tensor([  464,  5156,   318, 13774,   780,   673,   338,   587,  7165,  1263,
          290,  1263,   290,  1263,   290,  1263,    13,   843,  1263, 10963],
       device='cuda:0')
The baby is crying because she's been crazy big and big and big and big. And big monsters
!! tensor([  464,  5156,   318, 13774,   780,   673,   338,   587,  7165,  1263,
          290,  1263,   290,  1263,   290,  1263,   290,  1263,    13,  1263],
       device='cuda:0')
The baby is crying because she's been crazy big and big and big and big and big. big
------------------------------- Captured stderr --------------------------------
2022-01-20 12:25:52.703100: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-01-20 12:25:52.703133: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
=============================== warnings summary ===============================
../../../opt/conda/envs/ACW/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22
  /opt/conda/envs/ACW/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
    import imp

-- Docs: https://docs.pytest.org/en/stable/warnings.html
=========================== short test summary info ============================
ERROR tests/test.py - assert False
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
========================= 1 warning, 1 error in 22.90s =========================
