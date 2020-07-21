import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from transformers import theseus


@require_torch
class TheseusTest(unittest.TestCase):
    # These tests have a very small probability to fail even when the code is correct.
    # In this case, please re-run the test.
    def test_theseus_module(self):
        module_a = torch.nn.Linear(1, 1)
        module_b = torch.nn.Linear(1, 2)
        theseus_module = theseus.TheseusModule(predecessor=module_a, successor=module_b)

        for _ in range(10):
            self.assertEqual(theseus_module.sample_and_pass(), module_a)
        theseus_module.set_replacing_rate(1)
        for _ in range(10):
            self.assertEqual(theseus_module.sample_and_pass(), module_b)
        theseus_module.set_replacing_rate(0.5)

        prd_check, scc_check = False, False
        for _ in range(100):
            if theseus_module.sample_and_pass() == module_a:
                prd_check = True
            elif theseus_module.sample_and_pass() == module_b:
                scc_check = True
        self.assertTrue(prd_check)
        self.assertTrue(scc_check)

    def test_theseus_list_one_to_one(self):
        prd_module_list = torch.nn.ModuleList()
        scc_module_list = torch.nn.ModuleList()
        theseus_module_list = theseus.TheseusList()
        for i in range(10):
            prd_module = torch.nn.Linear(1, i)
            scc_module = torch.nn.Linear(2, i)
            theseus_module = theseus.TheseusModule(predecessor=prd_module, successor=scc_module)
            prd_module_list.append(prd_module)
            scc_module_list.append(scc_module)
            theseus_module_list.append(theseus_module)
        for _ in range(10):
            sampled_list = theseus_module_list.sample_and_pass()
            self.assertEqual(len(sampled_list), 10)
            for i in range(10):
                self.assertEqual(prd_module_list[i], sampled_list[i])

        theseus_module_list.set_replacing_rate(1)
        for _ in range(10):
            sampled_list = theseus_module_list.sample_and_pass()
            self.assertEqual(len(sampled_list), 10)
            for i in range(10):
                self.assertEqual(scc_module_list[i], sampled_list[i])

        theseus_module_list.set_replacing_rate(0.5)
        prd_num, scc_num = 0, 0
        for _ in range(10):
            sampled_list = theseus_module_list.sample_and_pass()
            self.assertEqual(len(sampled_list), 10)
            for i in range(10):
                if scc_module_list[i] == sampled_list[i]:
                    scc_num += 1
                elif prd_module_list[i] == sampled_list[i]:
                    prd_num += 1
        self.assertLess(scc_num / (prd_num + scc_num), 1)
        self.assertGreater(scc_num / (prd_num + scc_num), 0)

    def test_theseus_list_many_to_one(self):
        scc_module_list = torch.nn.ModuleList()
        theseus_module_list = theseus.TheseusList()
        for i in range(10):
            prd_module = torch.nn.ModuleList([torch.nn.Linear(1, i), torch.nn.Linear(2, i)])
            scc_module = torch.nn.Linear(3, i)
            theseus_module = theseus.TheseusModule(predecessor=prd_module, successor=scc_module)
            scc_module_list.append(scc_module)
            theseus_module_list.append(theseus_module)
        for _ in range(10):
            sampled_list = theseus_module_list.sample_and_pass()
            self.assertEqual(len(sampled_list), 20)

        theseus_module_list.set_replacing_rate(1)
        for _ in range(10):
            sampled_list = theseus_module_list.sample_and_pass()
            self.assertEqual(len(sampled_list), 10)
            for i in range(10):
                self.assertEqual(scc_module_list[i], sampled_list[i])

        theseus_module_list.set_replacing_rate(0.5)
        len_sum = 0
        for _ in range(100):
            sampled_list = theseus_module_list.sample_and_pass()
            len_sum += len(sampled_list)
        len_avg = len_sum / 100
        self.assertGreater(len_avg, 10)
        self.assertLess(len_avg, 20)

    def test_layerdrop_list(self):
        layers = torch.nn.ModuleList([torch.nn.Linear(1, i) for i in range(12)])
        drop_layers = theseus.LayerDropList.from_module_list(layers, 0)
        for _ in range(10):
            self.assertEqual(len(drop_layers.sample_and_pass()), 0)
        drop_layers.set_replacing_rate(1)
        for _ in range(10):
            self.assertEqual(len(drop_layers.sample_and_pass()), 12)
        drop_layers.set_replacing_rate(0.5)
        total_layers = 0
        for _ in range(100):
            total_layers += len(drop_layers.sample_and_pass())
        avg_layers = total_layers / 100
        self.assertLess(avg_layers, 12)
        self.assertGreater(avg_layers, 0)

    def test_mixout_list(self):
        layers = torch.nn.ModuleList([torch.nn.Linear(1, i) for i in range(12)])
        mixout_layers = theseus.MixoutList.from_module_list(layers, 0)
        for theseus_module in mixout_layers:
            self.assertNotEqual(theseus_module.predecessor, theseus_module.successor)
            self.assertEqual(theseus_module.predecessor.__class__, theseus_module.successor.__class__)
        for _ in range(10):
            self.assertEqual(len(mixout_layers.sample_and_pass()), 12)
        mixout_layers.set_replacing_rate(1)
        for _ in range(10):
            self.assertEqual(len(mixout_layers.sample_and_pass()), 12)
