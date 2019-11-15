import torch


def is_empty(x):
    if x is None:
        return True
    if isinstance(x, tuple) or isinstance(x, list):
        return all([is_empty(x_i) for x_i in x])
    if isinstance(x, State):
        return all([is_empty(getattr(x, s))
                    for s in State.__slots__ if s is not 'batch_first'])
    return False


class State(object):
    __slots__ = ['batch_first', 'hidden', 'inputs', 'outputs', 'context',
                 'attention', 'attention_score', 'mask', 'other']

    def __init__(self, hidden=None, inputs=None, outputs=None, context=None, attention=None,
                 attention_score=None, mask=None, batch_first=False, other=None):
        self.hidden = hidden
        self.outputs = outputs
        self.inputs = inputs
        self.context = context
        self.attention = attention
        self.mask = mask
        self.batch_first = batch_first
        self.attention_score = attention_score
        self.other=other
    def __select_state(self, state, i, type_state='hidden'):
        if isinstance(state, tuple):
            return tuple(self.__select_state(s, i, type_state) for s in state)
        elif torch.is_tensor(state):
            if type_state == 'hidden':
                batch_dim = 0 if state.dim() < 3 else 1
            else:
                batch_dim = 0 if self.batch_first else 1
            if state.size(batch_dim) > i:
                return state.narrow(batch_dim, i, 1)
            else:
                return None
        else:
            return state

    def __merge_states(self, state_list, type_state='hidden'):
        if state_list is None:
            return None
        if isinstance(state_list[0], State):
            return State().from_list(state_list)
        if isinstance(state_list[0], tuple):
            return tuple([self.__merge_states(s, type_state) for s in zip(*state_list)])
        else:
            if torch.is_tensor(state_list[0]):
                if type_state == 'hidden':
                    batch_dim = 0 if state_list[0].dim() < 3 else 1
                else:
                    batch_dim = 0 if self.batch_first else 1
                return torch.cat(state_list, batch_dim)
            else:
                assert state_list[1:] == state_list[:-1]  # all items are equal
                return state_list[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            state_list = [self[idx] for idx in range(
                index.start or 0, index.stop or len(self), index.step or 1)]
            return State().from_list(state_list)
        else:
            item = State()
            for s in self.__slots__:
                value = getattr(self, s, None)
                if isinstance(value, State):
                    selected_value = value[index]
                else:
                    selected_value = self.__select_state(value, index, s)
                setattr(item, s, selected_value)
            return item

    def as_list(self):
        i = 0
        out_list = []
        item = self.__getitem__(i)
        while not is_empty(item):
            out_list.append(item)
            i += 1
            item = self.__getitem__(i)
        return out_list

    def from_list(self, state_list):
        for s in self.__slots__:
            values = [getattr(item, s, None) for item in state_list]
            setattr(self, s, self.__merge_states(values, s))
        return self
