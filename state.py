from collections import defaultdict


class State:
    def __init__(self):
        self.cg_counter_A = defaultdict(lambda: defaultdict(int))
        self.cg_counter_B = defaultdict(lambda: defaultdict(int))

    def load_from_dicts(self, dict_A, dict_B):
        self.cg_counter_A = defaultdict(lambda: defaultdict(int))
        for key, subdict in dict_A.items():
            self.cg_counter_A[key] = defaultdict(int, subdict)

        self.cg_counter_B = defaultdict(lambda: defaultdict(int))
        for key, subdict in dict_B.items():
            self.cg_counter_B[key] = defaultdict(int, subdict)
